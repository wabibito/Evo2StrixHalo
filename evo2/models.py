import torch

from huggingface_hub import hf_hub_download
import yaml
from typing import List, Tuple, Dict, Union
from functools import partial

from vortex.model.generation import generate as vortex_generate
from vortex.model.model import StripedHyena
from vortex.model.tokenizer import CharLevelTokenizer
from vortex.model.utils import dotdict, print_rank_0, load_checkpoint

from evo2.scoring import score_sequences, score_sequences_rc
from evo2.utils import MODEL_NAMES, HF_MODEL_NAME_MAP, CONFIG_MAP

class Evo2:
    def __init__(self, model_name: str = MODEL_NAMES[1]):
        """
        Load an Evo 2 checkpoint.
        Automatically downloads checkpoint from huggingface if it does not exist.

        Evo 2 40b is too large to fit on a single H100 GPU, so needs multiple GPUs.
        Vortex automatically handles device placement on CUDA, and splits model across multiple GPUs if available.
        """

        if model_name not in MODEL_NAMES:
            raise ValueError(
                f'Invalid model name {model_name}. Should be one of: '
                f'{", ".join(MODEL_NAMES)}.'
            )

        config_path = CONFIG_MAP[model_name]
        
        # Load the model.
        self.model = self.load_evo2_model(model_name, config_path)
        self.tokenizer = CharLevelTokenizer(512)
            
    def forward(self, input_ids, return_embeddings=False, layer_names=None):
        """
        Forward pass with optional embedding extraction.
        
        Args:
            input_ids: Input token IDs
            return_embeddings: If True, returns embeddings from specified layers
            layer_names: List of layer names to extract embeddings from. Required if return_embeddings=True
            
        Returns:
            Tuple of (logits, embeddings_dict) if return_embeddings=True
            Tuple of (logits, None) otherwise
        """
        embeddings = {}
        handles = []
        
        if return_embeddings:
            if layer_names is None:
                raise ValueError("layer_names must be specified when return_embeddings=True")
                
            def hook_fn(layer_name):
                def hook(_, __, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    embeddings[layer_name] = output.detach()
                return hook
                
            # Register hooks for requested layers
            for name in layer_names:
                layer = self.model.get_submodule(name)
                handles.append(layer.register_forward_hook(hook_fn(name)))
        
        try:
            # Original forward pass
            with torch.no_grad():
                logits = self.model.forward(input_ids)
            
            if return_embeddings:
                return logits, embeddings
            return logits, None
            
        finally:
            for handle in handles:
                handle.remove()

    def score_sequences(
            self,
            seqs: List[str],
            batch_size: int = 1,
            prepend_bos: bool = False,
            reduce_method: str = 'mean',
            average_reverse_complement: bool = False,
    ) -> List[float]:
        scoring_func = partial(
            score_sequences_rc if average_reverse_complement else score_sequences,
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            prepend_bos=prepend_bos,
            reduce_method=reduce_method,
        )

        with torch.no_grad():
            try:
                scores = scoring_func(seqs)
            except Exception as e:
                raise RuntimeError(f"Error during sequence scoring: {str(e)}") from e

        return scores
    
    def generate(
        self,
        prompt_seqs: List[str],
        n_tokens: int = 500,
        temperature: float = 1.0,
        top_k: int = 4,
        top_p: float = 1.0,
        batched: bool = True,
        cached_generation: bool = True,
        verbose: int = 1,
        force_prompt_threshold: int = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Generate sequences from a list of prompts.

        If force_prompt_threshold is none, sets default assuming 1xH100 (evo2_7b) and 2xH100 (evo2_40b) to help avoid OOM errors.
        """
        if force_prompt_threshold is None:
            force_prompt_threshold = 8192 if '7b' in self.model.config.model_name else 5000

        with torch.no_grad():
            output = vortex_generate(
                prompt_seqs=prompt_seqs,
                model=self.model,
                tokenizer=self.tokenizer,
                n_tokens=n_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                batched=batched,
                cached_generation=cached_generation,
                verbose=verbose,
                force_prompt_threshold=force_prompt_threshold,
            )
            return output


    def load_evo2_model(
            self,
            model_name: str = MODEL_NAMES[1],
            config_path: str = 'vortex/configs/evo2_7b.yml',
    ):
        """
        Load HuggingFace checkpoint using StripedHyena 2.
        """
        hf_model_name = HF_MODEL_NAME_MAP[model_name]
        filename = f"{model_name}.pt"
        
        # First try normal download
        try:
            weights_path = hf_hub_download(
                repo_id=hf_model_name,
                filename=filename,
            )
        # If file is split, download and join parts
        except:
            weights_path = "/tmp/" + filename
            part_num = 0
            while True:
                try:
                    part = hf_hub_download(
                        repo_id=hf_model_name,
                        filename=f"{filename}.part{part_num}",
                    )
                    with open(weights_path, 'ab') as outfile:
                        with open(part, 'rb') as infile:
                            outfile.write(infile.read())
                    part_num += 1
                except:
                    break

        config = dotdict(yaml.load(open(config_path), Loader=yaml.FullLoader))
        model = StripedHyena(config)
        load_checkpoint(model, weights_path)

        print(f"Loaded model {model_name} from {weights_path}!")
        return model