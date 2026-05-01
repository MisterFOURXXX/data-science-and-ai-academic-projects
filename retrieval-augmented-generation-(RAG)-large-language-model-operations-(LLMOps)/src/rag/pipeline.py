import torch
import re
import os
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.rag.prompt_templates import create_prompt_text

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class OptimizedRAGPipeline:
    def __init__(self, config, vectorstore_path, model_path):
        self.config = config
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config.vectorstore.embedding_model,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )
        
        self.vectorstore = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=self.embedding_model,
            collection_name="stackoverflow_coding_train"
        )
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            print(f"Could not load tokenizer from {model_path}, loading from base model")
            self.tokenizer = AutoTokenizer.from_pretrained(config.model.base_model_name, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.model.quantization_4bit,
            bnb_4bit_quant_type=config.model.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=config.model.bnb_4bit_use_double_quant,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        try:
            self.model = PeftModel.from_pretrained(base_model, model_path)
            print(f"Loaded fine-tuned model from {model_path}")
        except Exception:
            print(f"No fine-tuned model found at {model_path}, using base model")
            self.model = base_model
        
        self.model.eval()
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": config.vectorstore.retrieval_k}
        )
        self.executor = ThreadPoolExecutor(max_workers=config.inference.num_workers)
    
    def retrieve_relevant_context(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        contexts = [f"[Score: {doc.metadata.get('score', 'N/A')}] {doc.page_content[:800]}" for doc in docs]
        return "\n\n---\n\n".join(contexts)
    
    def retrieve_batch_contexts(self, queries: list) -> list:
        return [self.retrieve_relevant_context(q) for q in queries]
    
    def batch_generate(self, prompts_texts: list) -> list:
        inputs = self.tokenizer(
            prompts_texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=1024
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.inference.max_new_tokens,
                temperature=self.config.inference.temperature,
                do_sample=True,
                top_p=self.config.inference.top_p,
                top_k=self.config.inference.top_k,
                repetition_penalty=self.config.inference.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        
        generated_responses = []
        for i, input_ids in enumerate(inputs.input_ids):
            generated_ids = outputs[i][len(input_ids):]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            solution_match = re.search(r'<solution>(.*?)</solution>', generated_text, re.DOTALL | re.IGNORECASE)
            if solution_match:
                solution_content = solution_match.group(1).strip()
                solution_content = re.sub(r'```\w*\n?|```', '', solution_content).strip()
                generated_responses.append(solution_content)
            else:
                generated_responses.append(generated_text)
        
        return generated_responses
    
    def get_answer(self, query: str) -> str:
        context = self.retrieve_relevant_context(query)
        prompt_text = create_prompt_text(query, context, self.tokenizer)
        return self.batch_generate([prompt_text])[0]
    
    def batch_query(self, queries: list) -> list:
        contexts = self.retrieve_batch_contexts(queries)
        prompt_texts = [create_prompt_text(q, c, self.tokenizer) for q, c in zip(queries, contexts)]
        return self.batch_generate(prompt_texts)