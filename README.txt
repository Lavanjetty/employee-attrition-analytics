# Employee Attrition Analytics — Deployment Bundle

This bundle was prepared from the uploaded Google Colab notebook.

## Important
Netlify serves the static website only. The Gradio/Python dashboard runs on a separate Python host. The current `gradio.live` URL is temporary and should not be used as the long-term embedded URL.

## A. Host the Gradio app on Hugging Face Spaces
1. Create a new Space at https://huggingface.co/new-space
2. Choose **Gradio** as the SDK.
3. Upload `app.py`, `requirements.txt`, and the required `HR_Attrition_dataset.csv` file to the root of the Space.
4. Wait for the build to finish and open the Space's App tab.
5. Test the dashboard and copy the permanent Space URL, for example `https://your-space-name.hf.space`.

The CSV is not included in this bundle because it was not present in the uploaded notebook file. Upload the same dataset CSV that you used in Colab. It must contain an `Attrition` column with Yes/No values and the features expected by the dashboard.

## B. Connect the Netlify website to the Space
1. Open `netlify-site/index.html`.
2. Find `https://YOUR-HUGGINGFACE-SPACE.hf.space`.
3. Replace it with your actual Space URL.
4. Deploy the `netlify-site` folder at https://app.netlify.com/drop.

## Files
- `app.py`: notebook code adapted for a hosted Python/Gradio environment (Colab upload magic removed).
- `requirements.txt`: Python package dependencies.
- `netlify-site/index.html`: responsive website frontend that embeds the hosted Gradio dashboard.

## Notes
- Keep the dataset private if it contains confidential employee information. Use only data you are authorized to publish.
- Free hosting may sleep when idle and may have resource limits.
