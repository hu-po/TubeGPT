import gradio as gr
import tubegpt.tube_openai
import tubegpt.tube_arxiv

tubegpt.tube_openai.set_key()


def foo(foo):
    print("foo")


def generate_texts(info, socials, title, hashtags):
    return f"""{title}{info}{socials}{hashtags}"


def generate_texts_title(prompt):
    return tubegpt.tube_openai.gpt_text(
        prompt=prompt,
        system=" ".join(
            [
                "You create titles for YouTube videos.",
                "Respond with a short title that best fits the description.",
                "Respond with the title only: no extra text or explanations.",
            ]
        ),
        temperature=0.6,
    )


def generate_texts_hashtags(prompt):
    return tubegpt.tube_openai.gpt_text(
        prompt=prompt,
        system=" ".join(
            [
                "You create hashtags for YouTube videos.",
                "Respond with up to 4 hashtags that match the user prompt.",
                "Respond with the hashtags only: no extra text or explanations.",
            ]
        ),
        temperature=0.6,
    )


SOCIALS = """
Like 👍. Comment 💬. Subscribe 🟥.

⌨️ GitHub
https://github.com/hu-po

🗨️ Discord
https://discord.gg/XKgVSxB6dE

📸 Instagram
http://instagram.com/gnocchibengal

"""

# Define the main GradIO UI
with gr.Blocks() as demo:
    num_tokens = gr.State(value=50)

    # Texts (Title, Descriptions, etc)
    texts_info = gr.State(value="")
    texts_socials = gr.State(value=SOCIALS)

    gr.HTML(
        """
        <center>
        <h1>TubeGPT</h1>
        </center>
    """
    )
    with gr.Tab("Texts"):
        gr_arxiv_link_textbox = gr.Textbox(
            value="https://arxiv.org/pdf/2106.04430.pdf",
            placeholder="Paste the arXiv link here",
            show_label=False,
            lines=1,
        )
        gr_arxiv_link_textbox.change(
            tubegpt.tube_arxiv.paper_blurb, gr_arxiv_link_textbox, texts_info
        )
        with gr.Row():
            gr_texts_title_button = gr.Button(value="Make Title")
            gr_texts_title_textbox = gr.Textbox()
            gr_texts_title_button.click(
                generate_texts_title,
                inputs=[texts_info],
                outputs=[gr_texts_title_textbox],
            )
        with gr.Row():
            gr_texts_hashtags_button = gr.Button(value="Make Hashtags")
            gr_texts_hashtags_textbox = gr.Textbox()
            gr_texts_hashtags_button.click(
                generate_texts_hashtags,
                inputs=[texts_info],
                outputs=[gr_texts_hashtags_textbox],
            )
        gr_generate_texts_button = gr.Button(value="Combine")
        gr_texts_textbox = gr.Textbox(label="Copy Paste into YouTube")
        gr_generate_texts_button.click(
            generate_texts,
            inputs=[
                texts_info,
                texts_socials,
                gr_texts_title_textbox,
                gr_texts_hashtags_textbox,
            ],
            outputs=[gr_texts_textbox],
        )

        # gr_model = gr.Dropdown(
        #     choices=["gpt-3.5-turbo", "gpt-4"],
        #     label="GPT Model behind conversation",
        #     value=STATE.model,
        # )
        # gr_max_tokens = gr.Slider(
        #     minimum=1, maximum=500, value=STATE.max_tokens, label="Max tokens", step=1
        # )
        # gr_temperature = gr.Slider(
        #     minimum=0.0,
        #     maximum=1.0,
        #     value=STATE.temperature,
        #     label="Temperature (randomness in conversation)",
        # )
    with gr.Tab("Thumbnail"):
        gr_generate_button = gr.Button(label="Generate")
        # gr_desc_gallery = gr.Gallery(
        #     label="Generated images", show_label=False, elem_id="gallery"
        # ).style(columns=[2], rows=[2], object_fit="contain", height="auto")
        # gr_generate_button.click(generate_desc, None, gr_desc_gallery)

    with gr.Tab("Keys"):
        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key here",
            show_label=False,
            lines=1,
            type="password",
        )
        openai_api_key_textbox.change(
            tubegpt.tube_openai.set_key, openai_api_key_textbox, None
        )
        # elevenlabs_api_key_textbox = gr.Textbox(
        #     placeholder="Paste your ElevenLabs API key here",
        #     show_label=False,
        #     lines=1,
        #     type="password",
        # )
        # elevenlabs_api_key_textbox.change(
        #     tubegpt.tube_elevenlabs.set_key, elevenlabs_api_key_textbox, None
        # )
        # replicate_api_key_textbox = gr.Textbox(
        #     placeholder="Paste your Replicate API key here",
        #     show_label=False,
        #     lines=1,
        #     type="password",
        # )
        # replicate_api_key_textbox.change(
        #     tubegpt.tube_replicate.set_key, replicate_api_key_textbox, None
        # )
        # discord_api_key_textbox = gr.Textbox(
        #     placeholder="Paste your Discord API key here",
        #     show_label=False,
        #     lines=1,
        #     type="password",
        # )
        # discord_api_key_textbox.change(
        #     tubegpt.tube_discord.set_key, discord_api_key_textbox, None
        # )
        # notion_api_key_textbox = gr.Textbox(
        #     placeholder="Paste your Notion API key here",
        #     show_label=False,
        #     lines=1,
        #     type="password",
        # )
        # notion_api_key_textbox.change(
        #     tubegpt.tube_notion.set_key, notion_api_key_textbox, None
        # )
        # notion_database_id_textbox = gr.Textbox(
        #     placeholder="Paste your Notion database ID here",
        #     show_label=False,
        #     lines=1,
        #     type="password",
        # )
        # notion_database_id_textbox.change(
        #     tubegpt.tube_notion.set_database_id, notion_database_id_textbox, None
        # )
        # google_api_key_textbox = gr.Textbox(
        #     placeholder="Paste your Google API key here",
        #     show_label=False,
        #     lines=1,
        #     type="password",
        # )
        # google_api_key_textbox.change(
        #     tubegpt.tube_google.set_key, google_api_key_textbox, None
        # )

    gr.HTML(
        """
        <center>
        Author: <a href="https://youtube.com/@hu-po">Hu Po</a>
        GitHub: <a href="https://github.com/hu-po/TubeGPT">TubeGPT</a>
        <br>
        <a href="https://huggingface.co/spaces/hu-po/speech2speech?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
        </center>
        """
    )

    # Key textboxes


if __name__ == "__main__":
    demo.launch()
