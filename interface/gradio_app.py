import gradio as gr
from pathlib import Path
import sys
import logging
import pandas as pd
from typing import Dict, List, Tuple
import os
import webbrowser

def setup_project_path() -> None:
    try:
        ROOT_PATH = Path(__file__).resolve().parents[1]
        if str(ROOT_PATH) not in sys.path:
            sys.path.insert(0, str(ROOT_PATH))
            logging.info(f"Added project root path: {ROOT_PATH}")
    except Exception as e:
        logging.error(f"Error setting up project path: {e}")
        raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

try:
    from codexmind.core.analyzer import RepoAnalyzer
    from codexmind.core.summarizer import CodeSummarizer
    from codexmind.llm.codellama import CodeLlamaLLM
    # Importar el generador de grafos con la ruta correcta
    from codexmind.core.grapher import CodeGraphGenerator
except ImportError as e:
    logging.error(f"Failed to import modules: {e}")
    raise

class CodexMindApp:
    def __init__(self):
        self.llm = CodeLlamaLLM()
        self.summarizer = CodeSummarizer(self.llm)
        self.repo_results: Dict = {}
        self.metrics_df = pd.DataFrame()
        self.graph_path = None
        logging.info("CodexMind application initialized")

    def analyze_repo(self, repo_path_str: str) -> Tuple[str, pd.DataFrame, str]:
        try:
            repo_path = Path(repo_path_str)
            if not repo_path.exists() or not repo_path.is_dir():
                return "❌ Ruta inválida", pd.DataFrame(), "No se ha generado ningún grafo"

            analyzer = RepoAnalyzer(repo_path)
            analyzer.config["max_workers"] = 1
            results = analyzer.analyze_repo()

            self.repo_results = results
            rows = []

            print(f"📂 Archivos detectados: {len(results)}")
            for filename, fa in results.items():
                print(f"  - {filename} | Funciones: {len(fa.functions)} | Clases: {len(fa.classes)}")

                try:
                    resumen = self.summarizer.generate_summary(fa, filename) if fa.functions or fa.classes else "No se pudo generar resumen."
                except Exception as e:
                    logging.error(f"Error al generar resumen para {filename}: {e}")
                    resumen = "❌ Error generando resumen"

                row = {
                    "Archivo": filename,
                    "Líneas totales": fa.total_lines,
                    "Código": fa.code_lines,
                    "Comentarios": fa.comment_lines,
                    "En blanco": fa.blank_lines,
                    "Funciones": len(fa.functions),
                    "Clases": len(fa.classes),
                    "Imports": ", ".join(fa.imports),
                    "Resumen IA": resumen
                }
                rows.append(row)

            self.metrics_df = pd.DataFrame(rows)
            
            # Generar el grafo
            html_path = self.generate_graph()
            
            if html_path:
                graph_message = f"✅ Grafo generado: {html_path}"
                # Convertir la ruta a URL para mostrar en el navegador
                file_url = Path(html_path).as_uri()
                logging.info(f"URL del grafo: {file_url}")
            else:
                graph_message = "❌ No se pudo generar el grafo"
                file_url = None
            
            return "✅ Análisis completado", self.metrics_df, graph_message

        except Exception as e:
            logging.error(f"Error inesperado en analyze_repo: {e}")
            return f"❌ Error: {e}", pd.DataFrame(), "Error al generar el grafo"

    def generate_graph(self) -> str:
        """
        Genera un grafo visual a partir de los resultados del análisis.
        
        Returns:
            Ruta al archivo HTML generado
        """
        try:
            if not self.repo_results:
                return None
                
            # Crear directorio para archivos de salida si no existe
            # Usar una carpeta relativa a la ubicación del script
            script_dir = Path(__file__).parent
            output_dir = script_dir / "output"
            output_dir.mkdir(exist_ok=True)
            
            # Generar nombre de archivo HTML basado en la fecha/hora
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = str(output_dir / f"code_graph_{timestamp}.html")
            
            # Crear el generador de grafos y generar el HTML
            graph_generator = CodeGraphGenerator(self.repo_results)
            self.graph_path = graph_generator.generate_html(html_path)
            
            logging.info(f"Grafo generado en: {self.graph_path}")
            return self.graph_path
            
        except Exception as e:
            logging.error(f"Error al generar el grafo: {e}")
            return None

    def open_graph_in_browser(self, graph_message: str) -> str:
        """
        Abre el grafo generado en el navegador predeterminado.
        
        Args:
            graph_message: Mensaje que incluye la ruta al grafo
            
        Returns:
            Mensaje de estado
        """
        try:
            if "✅ Grafo generado:" in graph_message:
                # Extraer la ruta del mensaje
                path = graph_message.replace("✅ Grafo generado:", "").strip()
                if os.path.exists(path):
                    # Abrir en el navegador
                    webbrowser.open(f'file://{os.path.abspath(path)}')
                    return "✅ Grafo abierto en el navegador"
                else:
                    return "❌ No se encontró el archivo del grafo"
            return "⚠️ No hay grafo para mostrar"
        except Exception as e:
            logging.error(f"Error al abrir el grafo: {e}")
            return f"❌ Error: {e}"

    def export_csv(self) -> str:
        try:
            if self.metrics_df.empty:
                return "⚠️ No hay datos para exportar."
            export_path = Path("export_codexmind.csv")
            self.metrics_df.to_csv(export_path, index=False)
            return f"✅ Exportado como {export_path.resolve()}"
        except Exception as e:
            return f"❌ Error al exportar: {e}"

    def create_interface(self):
        with gr.Blocks(title="CodexMind Repo View") as app:
            gr.Markdown("## 📊 CodexMind - Métricas del Repositorio y Resúmenes IA")

            repo_input = gr.Textbox(label="📁 Ruta del repositorio", placeholder="C:/ruta/a/tu/repositorio")
            analizar_btn = gr.Button("🔍 Analizar Repositorio")
            export_btn = gr.Button("📤 Exportar a CSV")
            estado = gr.Markdown()
            tabla = gr.Dataframe(label="📄 Métricas y Resumen por Archivo", interactive=False)
            
            # Estado del grafo
            grafo_estado = gr.Markdown()
            
            # Botón para abrir el grafo en el navegador
            open_graph_btn = gr.Button("🌐 Abrir Grafo en Navegador")
            open_graph_status = gr.Markdown()
            
            export_status = gr.Markdown()

            analizar_btn.click(
                fn=self.analyze_repo,
                inputs=repo_input,
                outputs=[estado, tabla, grafo_estado]
            )
            
            open_graph_btn.click(
                fn=self.open_graph_in_browser,
                inputs=grafo_estado,
                outputs=open_graph_status
            )

            export_btn.click(
                fn=self.export_csv,
                inputs=None,
                outputs=export_status
            )

        return app

def main():
    try:
        setup_project_path()
        app_instance = CodexMindApp()
        interface = app_instance.create_interface()
        interface.launch(inbrowser=True)
    except Exception as e:
        logging.error(f"❌ Error al iniciar CodexMindApp: {e}")

if __name__ == "__main__":
    main()