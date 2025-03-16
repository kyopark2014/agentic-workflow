# Graph 그리기

여기에서는 LangGraph로 만든 graph를 보여주는 방법에 대해 정리합니다.

## Streamlit에서 보여주기

아래와 같이 compile된 application인 app에서 get_graph()와 draw_mermaid_png로 이미지를 받은 후에 st.image로 화면에 표시할 수 있습니다.

```python
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

graphImage = Image(
    app.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API
    )
)
st.image(graphImage.data, caption="Graph", use_container_width=True)
```

## Graph를 파일로 저장하기

아래와 같이 graph에 대한 이미지를 파일로 저장할 수 있습니다.

```python
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

graphImage = Image(
    app.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API
    )
)

import PIL
import io
pimg = PIL.Image.open(io.BytesIO(graphImage.data))
pimg.save('graph-file.png')
```

## Reference

[How to visualize your graph](https://langchain-ai.github.io/langgraph/how-tos/visualization/)

[How to view or save an <IPython.core.display.Image object> using plain python (not ipython)](https://stackoverflow.com/questions/53424314/how-to-view-or-save-an-ipython-core-display-image-object-using-plain-python-n)

