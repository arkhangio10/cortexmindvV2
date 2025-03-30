import streamlit as st
from pathlib import Path
import sys

# Agregar el path raíz del proyecto
ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from codexmind.core.analyzer import RepoAnalyzer
from codexmind.core.summarizer import CodeSummarizer
from codexmind.llm.codellama import CodeLlamaLLM

# ---------- Configuración inicial ----------
st.set_page_config(page_title="CodexMind - Análisis de Código", layout="wide")
st.title("🧠 CodexMind: Panel Inteligente de Análisis de Código")

# ---------- Cargar modelo LLM ----------
@st.cache_resource
def load_llm():
    return CodeLlamaLLM()

llm = load_llm()
summarizer = CodeSummarizer(llm)

# ---------- Selección de ruta del repositorio ----------
repo_path = st.text_input("📁 Ruta del repositorio a analizar:", 
                          value="C:/Users/braya/Desktop/project_personal/proyectos_nuevos/CODEXMind/VERSION_2/repo_demo")

if repo_path and Path(repo_path).exists():
    repo_path = Path(repo_path)
    analyzer = RepoAnalyzer(repo_path)
    analyzer.config["max_workers"] = 1  # <- Evita errores de multiprocessing en Streamlit/Windows

    
    with st.spinner("🔍 Analizando código fuente..."):
        results = analyzer.analyze_repo()
        summary = analyzer.generate_summary()

    st.success("✅ Análisis completado")

    # ---------- Vista general del proyecto ----------
    st.subheader("📊 Resumen del Proyecto")
    st.metric("Total de Archivos", summary["total_files"])
    st.metric("Total de Funciones", summary["total_functions"])
    st.metric("Total de Clases", summary["total_classes"])
    st.metric("Promedio de líneas por archivo", f'{summary["avg_file_size"]:.1f}')
    st.metric("Ratio código/comentario", f'{summary["code_to_comment_ratio"]:.2f}')

    # ---------- Selector de archivo ----------
    st.subheader("📂 Archivos Analizados")
    file_selected = st.selectbox("Selecciona un archivo:", list(results.keys()))
    if file_selected:
        fa = results[file_selected]

        # Métricas del archivo
        st.markdown("### 📈 Métricas del Archivo")
        st.write({
            "Total de líneas": fa.total_lines,
            "Líneas de código": fa.code_lines,
            "Líneas en blanco": fa.blank_lines,
            "Líneas de comentario": fa.comment_lines,
            "Funciones": len(fa.functions),
            "Clases": len(fa.classes),
            "Imports": fa.imports
        })

        # Debug: mostrar el prompt generado
        st.markdown("### 🔍 Prompt generado")
        try:
            prompt_debug = summarizer._prompt_strategy.generate_prompt(
                file_selected,
                [cls.name for cls in fa.classes],
                [fn.name for fn in fa.functions],
                fa.imports
            )
            st.code(prompt_debug, language='text')
        except Exception as e:
            st.error(f"❌ Error generando prompt: {e}")

        # Resumen del archivo
        st.markdown("### 🧠 Resumen Generado por IA")
        try:
            resumen = summarizer.generate_summary(fa, file_selected)
            st.markdown(resumen)
        except Exception as e:
            st.error(f"❌ Error generando resumen con LLM:\n{e}")
else:
    st.warning("⚠️ Ingresa una ruta válida a un repositorio.")