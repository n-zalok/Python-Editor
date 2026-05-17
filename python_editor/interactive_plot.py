from pathlib import Path
import sys
import pandas as pd
from data_processing import split_by_developer
import umap
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, HoverTool, TapTool, CustomJS
from bokeh.transform import linear_cmap
from bokeh.palettes import RdBu11


ROOT_DIR = Path(__file__).resolve().parent.parent

emb_file = sys.argv[-1] 
df = pd.read_pickle(f"{ROOT_DIR}/data/{emb_file}")
train, test = split_by_developer(df, test_size=0.3, random_state=0)

embeddings = list(train["embedding"])
umap_embedder = umap.UMAP()
umap_emb = umap_embedder.fit_transform(embeddings)

train = pd.DataFrame({
    "x": umap_emb[:, 0],
    "y": umap_emb[:, 1],
    "NUM_CHARS": train["NUM_CHARS"],
    "score": train["pylint_score"],
    "text": train["text"]
})

source = ColumnDataSource(train[["x", "y", "NUM_CHARS", "score", "text"]])

color_mapping = linear_cmap(
        field_name="score",
        palette=RdBu11,
        low=train["score"].min(),
        high=train["score"].max()
)

hover = HoverTool(tooltips=[("NUM_CHARS", "@NUM_CHARS"), ("score", "@score")])
p = figure(title="UMAP embeddings", width=900, height=600)
p.add_tools(hover)

p.scatter("x", "y", source=source, color=color_mapping, size=6, fill_alpha=0.6, line_alpha=0)

# This script runs in the browser when a point is tapped
callback_js = CustomJS(args=dict(source=source), code="""
    const indices = source.selected.indices;
    if (indices.length > 0) {
        const index = indices[0];
        
        // Pull data from the source
        const score = source.data['score'][index];
        const text = source.data['text'][index];
        
        const newTab = window.open();
        
        // Construct a simple HTML layout
        const htmlContent = `
            <html>
            <head>
                <title>Data Point: ${index}</title>
                <style>
                    body { font-family: sans-serif; padding: 20px; line-height: 1.5; color: #333; }
                    .header { background: #f4f4f4; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #2196F3; }
                    pre { background: #272822; color: #f8f8f2; padding: 15px; border-radius: 5px; overflow: auto; }
                    b { color: #000; }
                </style>
            </head>
            <body>
                <div class="header">
                    <strong>Index (ID):</strong> ${index} <br>
                    <strong>Pylint Score:</strong> ${score.toFixed(2)}
                </div>
                <hr>
                <pre>${text}</pre>
            </body>
            </html>
        `;
        
        newTab.document.write(htmlContent);
        newTab.document.close();
    }
""")

# 3. Attach the JS callback to the selection event
source.selected.js_on_change('indices', callback_js)
p.add_tools(TapTool())

curdoc().add_root(p)