

from typing import Optional, Dict, Any, List, Union
from llama_cpp import Llama
from codexmind.llm.base import BaseLLM
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class CodeLlamaLLM(BaseLLM):
    def __init__(
        self,
        model_path: str = "C:/Users/braya/Desktop/project_personal/proyectos_nuevos/CODEXMind/VERSION_2/models/codellama-7b-q4_K_M.gguf",
        model_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Inicializa el modelo CodeLlama cuantizado en GGUF.

        Args:
            model_path: Ruta al archivo del modelo
            model_config: Configuración opcional del modelo
            **kwargs: Parámetros para la clase base (cache_dir, retries, etc.)
        """
        super().__init__(**kwargs)

        self.model_path = model_path
        self.model_config = model_config or {}
        self.model = None
        self.logger = logging.getLogger(self.__class__.__name__)

        self.load_model()

    def load_model(self) -> None:
        """Carga el modelo CodeLlama con configuración personalizada."""
        try:
            default_config = {
                'n_ctx': 2048,
                'n_gpu_layers': 16,
                'n_threads': 4,
                'verbose': False
            }
            config = {**default_config, **self.model_config}

            self.model = Llama(
                model_path=self.model_path,
                **config
            )
            self.logger.info(f"✅ Modelo CodeLlama cargado: {self.model_path}")
        except Exception as e:
            self.logger.error(f"❌ Error cargando modelo: {e}")
            raise

    def unload_model(self) -> None:
        """Libera recursos del modelo."""
        if self.model:
            del self.model
            self.model = None
            self.logger.info("🧹 Modelo CodeLlama descargado")

    def tokenize(self, text: str) -> List[Union[int, str]]:
        """Tokenización específica usando el modelo Llama."""
        try:
            return self.model.tokenize(text.encode('utf-8'))
        except Exception as e:
            self.logger.warning(f"⚠️ Error en tokenización: {e}")
            return super().tokenize(text)

    def ask(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Genera respuesta utilizando el modelo CodeLlama.

        Args:
            prompt: Texto de entrada
            max_new_tokens: Máximo de tokens a generar
            additional_params: Parámetros adicionales para generación

        Returns:
            Texto generado por el modelo
        """
        if not self.model:
            self.load_model()

        try:
            default_gen_params = {
                'max_tokens': max_new_tokens,
                'stop': ["</s>"],
                'temperature': 0.7,
                'top_p': 0.9,
                'echo': False
            }
            gen_params = {**default_gen_params, **(additional_params or {})}

            response = self.model(prompt, **gen_params)
            return response["choices"][0]["text"].strip()

        except Exception as e:
            self.logger.error(f"❌ Error generando respuesta: {e}")
            raise RuntimeError(f"Error generando respuesta: {e}")

if __name__ == "__main__":
    print("🔍 Iniciando prueba con CodeLlama LLM...\n")
    
    try:
        llm = CodeLlamaLLM()
        prompt = "Explica brevemente qué es Python y para qué se utiliza."
        respuesta = llm.ask(prompt, max_new_tokens=150)

        print("\n📤 Prompt:")
        print(prompt)
        print("\n📥 Respuesta generada:")
        print(respuesta)
    except Exception as e:
        print(f"❌ Error al ejecutar CodeLlama: {e}")
