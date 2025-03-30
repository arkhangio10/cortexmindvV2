import os
import json
import hashlib
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import logging
import time

class BaseLLM(ABC):
    def __init__(
        self, 
        cache_dir: Optional[str] = None, 
        max_cache_size: int = 1000,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Inicializa el LLM base con configuraciones de caché y reintento.
        
        Args:
            cache_dir: Directorio para almacenar caché de respuestas
            max_cache_size: Número máximo de entradas en caché
            max_retries: Número máximo de reintentos en caso de fallo
            retry_delay: Tiempo de espera entre reintentos
        """
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".llm_cache")
        self.max_cache_size = max_cache_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Asegurar directorio de caché
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
    
    def _generate_cache_key(self, prompt: str, params: Dict[str, Any]) -> str:
        """
        Genera una clave única para cachear basada en prompt y parámetros.
        
        Args:
            prompt: Texto de entrada
            params: Parámetros de generación
        
        Returns:
            Clave hash única
        """
        key_data = json.dumps({
            'prompt': prompt, 
            'params': sorted(params.items())
        }, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """
        Obtiene respuesta de caché si existe.
        
        Args:
            cache_key: Clave de caché
        
        Returns:
            Respuesta cacheada o None
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_entry = json.load(f)
                    return cache_entry['response']
            except Exception as e:
                self.logger.warning(f"Error leyendo caché: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, response: str):
        """
        Guarda respuesta en caché.
        
        Args:
            cache_key: Clave de caché
            response: Respuesta a guardar
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'response': response
                }, f)
        except Exception as e:
            self.logger.warning(f"Error guardando caché: {e}")
    
    def ask_with_retry(
        self, 
        prompt: str, 
        max_new_tokens: int = 200, 
        additional_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Método de generación con caché y reintento.
        
        Args:
            prompt: Texto de entrada
            max_new_tokens: Máximo de tokens a generar
            additional_params: Parámetros adicionales
        
        Returns:
            Respuesta generada
        """
        params = additional_params or {}
        cache_key = self._generate_cache_key(prompt, params)
        
        # Verificar caché primero
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response
        
        # Reintentar generación
        for attempt in range(self.max_retries):
            try:
                response = self.ask(
                    prompt, 
                    max_new_tokens=max_new_tokens, 
                    additional_params=params
                )
                
                # Guardar en caché
                self._save_to_cache(cache_key, response)
                return response
            
            except Exception as e:
                self.logger.warning(f"Intento {attempt + 1} fallido: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Backoff exponencial
                else:
                    raise
    
    def tokenize(self, text: str) -> List[str]:
        """
        Método para tokenización básica.
        
        Args:
            text: Texto a tokenizar
        
        Returns:
            Lista de tokens
        """
        return text.split()
    
    def count_tokens(self, text: str) -> int:
        """
        Cuenta el número de tokens en un texto.
        
        Args:
            text: Texto a contar
        
        Returns:
            Número de tokens
        """
        return len(self.tokenize(text))

    @abstractmethod
    def ask(
        self, 
        prompt: str, 
        max_new_tokens: int = 200, 
        additional_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Método abstracto para generar respuestas."""
        pass