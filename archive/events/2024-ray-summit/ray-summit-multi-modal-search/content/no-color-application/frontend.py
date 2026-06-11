from typing import Optional
import gradio as gr
from ray.serve.gradio_integrations import GradioServer
import requests

ANYSCALE_BACKEND_SERVICE_URL = "http://localhost:8000/backend"


def filter_products_legacy(
    text_query: Optional[str],
    min_price: int,
    max_price: int,
    min_rating: float,
    num_results: int,
) -> list[tuple[str, str]]:
    response = requests.get(
        f"{ANYSCALE_BACKEND_SERVICE_URL}/legacy",
        params={
            "text_search": text_query or "",
            "min_price": min_price,
            "max_price": max_price,
            "min_rating": min_rating,
            "num_results": num_results,
        },
    )
    return response.json()


def filter_products_with_ai(
    text_search: Optional[str],
    min_price: int,
    max_price: int,
    min_rating: float,
    categories: list[str],
    colors: list[str],
    seasons: list[str],
    num_results: int,
    search_type: list[str],
    embedding_column: str,
):
    params = {
        "text_search": text_search or "",
        "min_price": min_price,
        "max_price": max_price,
        "min_rating": min_rating,
        "num_results": num_results,
        "embedding_column": embedding_column,
    }
    body = {
        "categories": categories,
        "colors": colors,
        "seasons": seasons,
        "search_type": search_type,
    }

    response = requests.get(
        f"{ANYSCALE_BACKEND_SERVICE_URL}/ai_enabled",
        params=params,
        json=body,
    )
    results = response.json()

    return results


def build_interface():
    price_min = 0
    price_max = 100_000

    # Get rating range
    rating_min = 0
    rating_max = 5

    # Gradio Interface
    with gr.Blocks(
        # theme="shivi/calm_foam",
        title="Multi-modal search",
    ) as iface:
        with gr.Tab(label="Legacy Search"):
            with gr.Row():
                with gr.Column(scale=1):
                    keywords_component = gr.Textbox(label="Keywords")
                    min_price_component = gr.Slider(
                        price_min, price_max, label="Min Price", value=price_min
                    )
                    max_price_component = gr.Slider(
                        price_min, price_max, label="Max Price", value=price_max
                    )
                    min_rating_component = gr.Slider(
                        rating_min, rating_max, step=0.25, label="Min Rating"
                    )
                    max_num_results_component = gr.Slider(
                        1, 100, step=1, label="Max Results", value=20
                    )
                    filter_button_component = gr.Button("Filter")
                with gr.Column(scale=3):
                    gallery = gr.Gallery(
                        label="Filtered Products",
                        columns=3,
                        height=800,
                    )
            inputs = [
                keywords_component,
                min_price_component,
                max_price_component,
                min_rating_component,
                max_num_results_component,
            ]
            filter_button_component.click(
                filter_products_legacy, inputs=inputs, outputs=gallery
            )
            iface.load(
                filter_products_legacy,
                inputs=inputs,
                outputs=gallery,
            )

        with gr.Tab(label="AI enabled search"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_component = gr.Textbox(label="Text Search")
                    min_price_component = gr.Slider(
                        price_min, price_max, label="Min Price", value=price_min
                    )
                    max_price_component = gr.Slider(
                        price_min, price_max, label="Max Price", value=price_max
                    )

                    min_rating_component = gr.Slider(
                        rating_min, rating_max, step=0.25, label="Min Rating"
                    )
                    category_component = gr.CheckboxGroup(
                        ["Tops", "Bottoms", "Dresses", "Footwear", "Accessories"],
                        label="Category",
                        value=[
                            "Tops",
                            "Bottoms",
                            "Dresses",
                            "Footwear",
                            "Accessories",
                        ],
                    )
                    season_component = gr.CheckboxGroup(
                        ["Summer", "Winter", "Spring", "Fall"],
                        label="Season",
                        value=[
                            "Summer",
                            "Winter",
                            "Spring",
                            "Fall",
                        ],
                    )
                    color_component = gr.CheckboxGroup(
                        [
                            "Red",
                            "Blue",
                            "Green",
                            "Yellow",
                            "Black",
                            "White",
                            "Pink",
                            "Purple",
                            "Orange",
                            "Brown",
                            "Grey",
                        ],
                        label="Color",
                        value=[
                            "Red",
                            "Blue",
                            "Green",
                            "Yellow",
                            "Black",
                            "White",
                            "Pink",
                            "Purple",
                            "Orange",
                            "Brown",
                            "Grey",
                        ],
                    )
                    max_num_results_component = gr.Slider(
                        1, 100, step=1, label="Max Results", value=20
                    )

                    # add an engine advanced options
                    with gr.Accordion(label="Advanced Engine Options"):
                        # checkbox for type of search - lexical and/or vector
                        search_type_component = gr.CheckboxGroup(
                            ["Lexical", "Vector"],
                            label="Search Type",
                            value=["Lexical", "Vector"],
                        )
                        # dropdwon for embedding column - name or description
                        embedding_column_component = gr.Dropdown(
                            ["name", "description"],
                            label="Embedding Column",
                            value="description",
                        )

                    filter_button_component = gr.Button("Filter")
                with gr.Column(scale=3):
                    gallery = gr.Gallery(
                        label="Filtered Products",
                        columns=3,
                        height=800,
                    )
            inputs = [
                text_component,
                min_price_component,
                max_price_component,
                min_rating_component,
                category_component,
                color_component,
                season_component,
                max_num_results_component,
                search_type_component,
                embedding_column_component,
            ]

            filter_button_component.click(
                filter_products_with_ai,
                inputs=inputs,
                outputs=gallery,
            )
            iface.load(
                filter_products_with_ai,
                inputs=inputs,
                outputs=gallery,
            )

    return iface


app = GradioServer.options(ray_actor_options={"num_cpus": 1}).bind(build_interface)
