import os
import logging
import typing as t
from abc import ABC, abstractmethod

from mlflow.tracking import MlflowClient

# export .md to .pdf
import markdown
from weasyprint import HTML
import tempfile

# resolving files in templates module within package
from importlib.abc import Traversable
from importlib.resources import files

# relative imports in 'reporting' module
from ..sections import register_sections
from ..sections.registry import SectionRegistry
from ..utils import utils
from ..utils.formatting import Formatting
from ..utils.types import ModelCardConfig

LOGGER = logging.getLogger(__name__)
C = t.TypeVar("C", bound=ModelCardConfig)


class ModelCard(t.Generic[C], ABC):
    DEFAULT_ASSETS_FOLDER = "card_assets"
    TEMPLATE_FILENAME = "model-card-TEMPLATE.md"
    LOGO_FILENAME = "logo.png"

    def __init__(
        self,
        config: C,
        catalog: str,
        model_name: str,
        assets_path: t.Optional[str] = None,
        mlflow_client: t.Optional[MlflowClient] = None,
    ):
        """
        Initializes the ModelCard object with the given config and the model name
        in unity catalog. If assets_path is not provided, the default assets folder is used.
        """
        self.cfg = config
        self.catalog = catalog
        self.model_name = model_name
        self.uc_model_name = f"{catalog}.{self.cfg.institution_id}_gold.{model_name}"
        LOGGER.info("Initializing ModelCard for model: %s", self.uc_model_name)

        self.client = mlflow_client or MlflowClient()
        self.section_registry = SectionRegistry()
        self.format = Formatting()
        self.context: dict[str, t.Any] = {}

        self.assets_folder = assets_path or self.DEFAULT_ASSETS_FOLDER
        self.output_path = self._build_output_path()
        self.template_path = self._resolve(
            "edvise.reporting.template", self.TEMPLATE_FILENAME
        )
        self.logo_path = self._resolve(
            "edvise.reporting.template.assets", self.LOGO_FILENAME
        )

    def build(self):
        """
        Builds the model card by performing the following steps:
        1. Loads the MLflow model.
        2. Finds the model version from the MLflow client based on the run ID.
        3. Extracts the training data from the MLflow run.
        4. Registers all sections in the section registry.
        5. Collects all metadata for the model card.
        6. Renders the model card using the template and context.
        """
        self.load_model()
        self.find_model_version()
        self.extract_training_data()
        self._register_sections()
        self.collect_metadata()
        self.render()

    @abstractmethod
    def load_model(self):
        """
        Loads the model from MLflow.

        Subclasses must implement this to handle framework-specific model loading
        (e.g., H2O, sklearn, etc.) and assign self.model, self.run_id, and
        self.experiment_id.
        """
        pass

    def find_model_version(self):
        """
        Retrieves the model version from the MLflow client based on the run ID.
        """
        try:
            versions = self.client.search_model_versions(f"name='{self.uc_model_name}'")
            for v in versions:
                if v.run_id == self.run_id:
                    self.context["version_number"] = v.version
                    LOGGER.info(f"Model Version = {self.context['version_number']}")
                    return
            LOGGER.warning(f"Unable to find model version for run id: {self.run_id}")
            self.context["version_number"] = None
        except Exception as e:
            LOGGER.error(
                f"Error retrieving model version for run id {self.run_id}: {e}"
            )
            self.context["version_number"] = None

    @abstractmethod
    def extract_training_data(self):
        """
        Extracts the training data from the MLflow run.

        Subclasses must implement this to handle framework-specific data extraction
        and populate self.modeling_data, self.training_data, and relevant context fields.
        """
        pass

    def collect_metadata(self):
        """
        Gathers all metadata for the model card. All of this data is dynamic and will
        depend on the institution and model. This calls functions that retrieves & downloads
        mlflow artifacts and also retrieves config information.
        """
        metadata_functions = [
            self.get_basic_context,
            self.get_feature_metadata,
            self.get_model_plots,
            self.section_registry.render_all,
        ]

        for func in metadata_functions:
            LOGGER.info(f"Updating context from {func.__name__}()")
            self.context.update(func())

    def get_basic_context(self) -> dict[str, str]:
        """
        Collects "basic" context which instantiates the DataKind logo, the
        institution name, and the current year.

        Returns:
            A dictionary with the keys as the variable names that will be called
            dynamically in template with values for each variable.
        """
        return {
            "logo": utils.download_static_asset(
                description="Logo",
                static_path=self.logo_path,
                local_folder=self.assets_folder,
            )
            or "",
            "institution_name": self.cfg.institution_name,
        }

    @abstractmethod
    def get_feature_metadata(self) -> dict[str, str]:
        """
        Collects feature count and feature selection metadata.

        Subclasses must implement this to extract framework-specific feature information
        (e.g., from sklearn pipeline or H2O model) and combine it with config data.

        Returns:
            A dictionary with feature metadata for the template.
        """
        pass

    @abstractmethod
    def get_model_plots(self) -> dict[str, str]:
        """
        Collects model plots from the MLflow run and downloads them locally.

        Subclasses must implement this to specify which plot artifacts to download
        and how to format them.

        Returns:
            A dictionary with plot names as keys and inline HTML as values.
        """
        pass

    def render(self):
        """
        Renders the model card using the template and context data.
        """
        with open(self.template_path, "r") as file:
            template = file.read()
        filled = template.format(**self.context)
        with open(self.output_path, "w") as file:
            file.write(filled)
        LOGGER.info(f"✅ Model card generated at {self.output_path}")

    def reload_card(self):
        """
        Reloads Markdown model card post user editing after rendering.
        This offers flexibility in case user wants to utilize this class
        as a base and then makes edits in markdown before exporting as a PDF.
        """
        # Read the Markdown output
        with open(self.output_path, "r") as f:
            self.md_content = f.read()
        LOGGER.info("Reloaded model card content")

    def style_card(self):
        """
        Styles card using CSS.
        """
        # Build a Markdown renderer with the right extensions
        md = markdown.Markdown(
            extensions=[
                "extra",  # code, tables, etc.
                "tables",
                "sane_lists",
                "attr_list",  # {#id} and {.class} on headings
                "toc",  # [TOC] + internal anchors
                "smarty",
            ],
            extension_configs={
                "toc": {
                    "permalink": False,
                    "toc_depth": "2-6",
                }
            },
        )

        # Convert Markdown text → HTML string
        html_body = md.convert(
            self.md_content
        )  # ← this is a str, not a Markdown object

        # Load CSS from external file
        css_path = self._resolve("edvise.reporting.template.styles", "model_card.css")
        with open(css_path, "r") as f:
            style = f"<style>\n{f.read()}\n</style>"

        # Prepend CSS to HTML
        self.html_content = style + html_body
        LOGGER.info("Applied CSS styling")

    def export_to_pdf(self):
        """
        Exports markdown to weasyprint with CSS styling.
        """
        self.style_card()

        # Images are relative to the generated markdown/html location
        base_path = os.path.dirname(self.output_path) or "."
        self.pdf_path = self.output_path.replace(".md", ".pdf")

        try:
            HTML(string=self.html_content, base_url=base_path).write_pdf(self.pdf_path)
            LOGGER.info(f"✅ PDF model card saved to {self.pdf_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to create PDF: {e}")

        utils.save_card_to_gold_volume(
            filename=self.pdf_path,
            catalog=self.catalog,
            institution_id=self.cfg.institution_id,
        )

    def _build_output_path(self) -> str:
        """
        Builds the output path for the model card.
        """
        out_dir = os.path.join(tempfile.gettempdir(), "model_cards")
        os.makedirs(out_dir, exist_ok=True)
        filename = f"model-card-{self.model_name}.md"
        return os.path.join(out_dir, filename)

    def _register_sections(self):
        """
        Registers all sections in the section registry.
        """
        register_sections(self, self.section_registry)

    def _resolve(self, package: str, filename: str) -> Traversable:
        """
        Resolves files using importlib. Importlib is necessary since
        the file exists within the SST package itself.
        """
        return files(package).joinpath(filename)
