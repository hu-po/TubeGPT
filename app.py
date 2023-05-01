import gradio as gr

def foo(foo):
    print("foo")

# Define the main GradIO UI
with gr.Blocks() as demo:
    num_tokens = gr.State(value=50)

    gr.HTML(
        """
    <center>
    <h1>TubeGPT</h1>
    </center>"""
    )
    with gr.Tab("Title and Description"):
        pass
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
    with gr.Tab("Keys"):
        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key here",
            show_label=False,
            lines=1,
            type="password",
        )
        elevenlabs_api_key_textbox = gr.Textbox(
            placeholder="Paste your ElevenLabs API key here",
            show_label=False,
            lines=1,
            type="password",
        )
        replicate_api_key_textbox = gr.Textbox(
            placeholder="Paste your Replicate API key here",
            show_label=False,
            lines=1,
            type="password",
        )
        discord_api_key_textbox = gr.Textbox(
            placeholder="Paste your Discord API key here",
            show_label=False,
            lines=1,
            type="password",
        )
        notion_api_key_textbox = gr.Textbox(
            placeholder="Paste your Notion API key here",
            show_label=False,
            lines=1,
            type="password",
        )
        notion_database_id_textbox = gr.Textbox(
            placeholder="Paste your Notion database ID here",
            show_label=False,
            lines=1,
            type="password",
        )
        google_api_key_textbox = gr.Textbox(
            placeholder="Paste your Google API key here",
            show_label=False,
            lines=1,
            type="password",
        )

    with gr.Tab("Thumbnail"):
        pass

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
    openai_api_key_textbox.change(foo, openai_api_key_textbox, None)

if __name__ == "__main__":
    demo.launch()
