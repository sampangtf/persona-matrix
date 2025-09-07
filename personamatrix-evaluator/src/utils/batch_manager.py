"""
Minimal Batch Manager
Submit, monitor, and retrieve results from Gemini, OpenAI, and Anthropic batch APIs
"""
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal

# Provider-specific imports with graceful fallbacks
try:
    import google.generativeai as genai
    from google import genai as google_genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

ProviderType = Literal["gemini", "openai", "anthropic"]


class BatchManager:
    """Minimal batch manager for all LLM providers."""
    
    def __init__(self, provider: ProviderType = "gemini"):
        self.provider = provider
        self._setup_client()
    
    def _setup_client(self):
        """Setup the appropriate client based on provider."""
        if self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("Install: pip install google-generativeai")
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment")
            self.client = google_genai.Client(api_key=api_key)
            
        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("Install: pip install openai")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.client = openai.OpenAI(api_key=api_key)
            
        elif self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Install: pip install anthropic")
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            self.client = anthropic.Anthropic(api_key=api_key)
    
    def submit_batch(self, batch_file: str, model: Optional[str] = None, job_name: Optional[str] = None) -> str:
        """Submit batch file to API."""
        if self.provider == "gemini":
            return self._submit_gemini(batch_file, model, job_name)
        elif self.provider == "openai":
            return self._submit_openai(batch_file, job_name)
        elif self.provider == "anthropic":
            return self._submit_anthropic(batch_file)
    
    def _submit_gemini(self, batch_file: str, model: Optional[str], job_name: Optional[str]) -> str:
        """Submit to Gemini Batch API."""
        model = model or "models/gemini-2.0-flash-exp"
        job_name = job_name or f"batch_{int(time.time())}"
        
        # Upload file
        uploaded_file = self.client.files.upload(
            file=batch_file,
            config=google_genai.types.UploadFileConfig(
                display_name=job_name,
                mime_type='application/jsonl'
            )
        )
        
        # Create batch job
        batch_job = self.client.batches.create(
            model=model,
            src=uploaded_file.name,
            config={'display_name': job_name}
        )
        
        print(f"✅ Submitted Gemini batch: {batch_job.name}")
        return batch_job.name
    
    def _submit_openai(self, batch_file: str, job_name: Optional[str]) -> str:
        """Submit to OpenAI Batch API."""
        # Upload file
        with open(batch_file, 'rb') as f:
            batch_input_file = self.client.files.create(file=f, purpose="batch")
        
        # Create batch
        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"description": job_name or "batch_job"}
        )
        
        print(f"✅ Submitted OpenAI batch: {batch.id}")
        return batch.id
    
    def _submit_anthropic(self, batch_file: str) -> str:
        """Submit to Anthropic Batch API."""
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
        
        batch = self.client.messages.batches.create(requests=batch_data["requests"])
        
        print(f"✅ Submitted Anthropic batch: {batch.id}")
        return batch.id
    
    def monitor_batch(self, job_id: str, poll_interval: int = 30) -> str:
        """Monitor batch until completion."""
        print(f"🔄 Monitoring {self.provider} batch: {job_id}")
        
        while True:
            status = self.get_status(job_id)
            print(f"📊 Status: {status}")
            
            if self._is_complete(status):
                print(f"✅ Batch completed with status: {status}")
                return status
            elif self._is_failed(status):
                print(f"❌ Batch failed with status: {status}")
                return status
                
            time.sleep(poll_interval)
    
    def get_status(self, job_id: str) -> str:
        """Get current batch status."""
        if self.provider == "gemini":
            batch_job = self.client.batches.get(name=job_id)
            return batch_job.state.name
        elif self.provider == "openai":
            batch = self.client.batches.retrieve(job_id)
            return batch.status
        elif self.provider == "anthropic":
            batch = self.client.messages.batches.retrieve(job_id)
            return batch.processing_status
    
    def _is_complete(self, status: str) -> bool:
        """Check if batch is complete."""
        if self.provider == "gemini":
            return status == "JOB_STATE_SUCCEEDED"
        elif self.provider == "openai":
            return status == "completed"
        elif self.provider == "anthropic":
            return status == "ended"
        return False
    
    def _is_failed(self, status: str) -> bool:
        """Check if batch failed."""
        if self.provider == "gemini":
            return status in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]
        elif self.provider == "openai":
            return status in ["failed", "expired", "cancelled"]
        elif self.provider == "anthropic":
            return status in ["canceled", "expired"]
        return False
    
    def download_results(self, job_id: str, output_file: str) -> str:
        """Download batch results."""
        if self.provider == "gemini":
            return self._download_gemini(job_id, output_file)
        elif self.provider == "openai":
            return self._download_openai(job_id, output_file)
        elif self.provider == "anthropic":
            return self._download_anthropic(job_id, output_file)
    
    def _download_gemini(self, job_id: str, output_file: str) -> str:
        """Download Gemini results."""
        batch_job = self.client.batches.get(name=job_id)
        result_file_name = batch_job.dest.file_name
        file_content = self.client.files.download(file=result_file_name)
        
        with open(output_file, 'wb') as f:
            f.write(file_content)
        
        print(f"📥 Downloaded Gemini results to: {output_file}")
        return output_file
    
    def _download_openai(self, job_id: str, output_file: str) -> str:
        """Download OpenAI results."""
        batch = self.client.batches.retrieve(job_id)
        file_response = self.client.files.content(batch.output_file_id)
        
        with open(output_file, 'wb') as f:
            f.write(file_response.content)
        
        print(f"📥 Downloaded OpenAI results to: {output_file}")
        return output_file
    
    def _download_anthropic(self, job_id: str, output_file: str) -> str:
        """Download Anthropic results."""
        batch = self.client.messages.batches.retrieve(job_id)
        
        # Stream results from URL
        import requests
        response = requests.get(batch.results_url, headers={
            "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
            "anthropic-version": "2023-06-01"
        })
        
        with open(output_file, 'w') as f:
            f.write(response.text)
        
        print(f"📥 Downloaded Anthropic results to: {output_file}")
        return output_file
    
    def process_results(self, results_file: str) -> Dict[str, Any]:
        """Parse and normalize results from all providers."""
        results = {}
        errors = []
        
        with open(results_file, 'r') as f:
            if self.provider == "anthropic":
                # Anthropic uses JSONL
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        custom_id = data.get("custom_id")
                        if data.get("result", {}).get("type") == "succeeded":
                            message = data["result"]["message"]
                            text = message["content"][0]["text"] if message.get("content") else ""
                            results[custom_id] = {"response_text": text, "full_response": data}
                        else:
                            errors.append({"custom_id": custom_id, "error": data.get("result", {})})
                    except json.JSONDecodeError as e:
                        errors.append({"line": line_num, "error": f"JSON decode error: {e}"})
            
            else:
                # Gemini and OpenAI use JSONL
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        
                        if self.provider == "gemini":
                            key = data.get("key", f"unknown_{line_num}")
                            response = data.get("response", {})
                            if "candidates" in response and response["candidates"]:
                                candidate = response["candidates"][0]
                                if "content" in candidate and "parts" in candidate["content"]:
                                    text = candidate["content"]["parts"][0].get("text", "")
                                    results[key] = {"response_text": text, "full_response": response}
                            else:
                                errors.append({"key": key, "error": "No candidates in response"})
                        
                        elif self.provider == "openai":
                            custom_id = data.get("custom_id")
                            if data.get("response"):
                                response = data["response"]["body"]
                                if response.get("choices"):
                                    text = response["choices"][0]["message"]["content"]
                                    results[custom_id] = {"response_text": text, "full_response": response}
                                else:
                                    errors.append({"custom_id": custom_id, "error": "No choices in response"})
                            else:
                                errors.append({"custom_id": custom_id, "error": data.get("error", "Unknown error")})
                    
                    except json.JSONDecodeError as e:
                        errors.append({"line": line_num, "error": f"JSON decode error: {e}"})
        
        print(f"📊 Processed {len(results)} responses, {len(errors)} errors")
        return {"results": results, "errors": errors, "total_processed": len(results) + len(errors)}


def quick_batch(batch_file: str, provider: ProviderType = "gemini", model: Optional[str] = None) -> Dict[str, Any]:
    """Quick batch processing - submit, monitor, download, and parse results."""
    manager = BatchManager(provider)
    
    # Submit
    job_id = manager.submit_batch(batch_file, model)
    
    # Monitor
    final_status = manager.monitor_batch(job_id)
    
    if not manager._is_complete(final_status):
        return {"status": "failed", "job_id": job_id, "final_status": final_status}
    
    # Download results
    output_file = f"results_{provider}_{int(time.time())}.jsonl"
    manager.download_results(job_id, output_file)
    
    # Process results
    parsed_results = manager.process_results(output_file)
    
    return {
        "status": "success",
        "job_id": job_id,
        "results_file": output_file,
        "results": parsed_results
    }
