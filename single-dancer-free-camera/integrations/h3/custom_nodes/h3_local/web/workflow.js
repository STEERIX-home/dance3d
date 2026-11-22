import { app } from "../../scripts/app.js";
app.registerExtension({
  name: "local.h3.character-swap",
  afterConfigureGraph() {
    if (!app.graph.extra?.h3_local_recipe) return;
    for (const node of app.graph._nodes) {
      if (node.type !== "CreateVideo") continue;
      const chroma = node.widgets?.find(widget => widget.name === "chroma_subsampling");
      // ComfyUI added this widget before codec; older presets put "none" in its slot.
      if (chroma?.value === "none") chroma.value = "auto";
    }
  },
  async setup() {
    const status = await fetch("/h3-local/status").then(r => r.json());
    const loaded = localStorage.getItem("h3-local-workflow-loaded");
    const loadWhenCanvasReady = async () => {
      if (!document.querySelector("canvas")) {
        setTimeout(loadWhenCanvasReady, 100);
        return;
      }
      await new Promise(requestAnimationFrame);
      if (!loaded) {
        const workflow = await fetch("/h3-local/workflow").then(r => r.json());
        await app.loadGraphData(workflow, true, true, "H3_Character_Swap_5090");
        localStorage.setItem("h3-local-workflow-loaded", "1");
      }
    };
    setTimeout(loadWhenCanvasReady, 100);
    document.title = "H3 캐릭터 스왑 · ComfyUI";
    if (!status.models_ready) {
      app.extensionManager.toast.add({severity:"warn",summary:"H3 모델 파일 준비 대기",detail:"UI와 워크플로는 설치되었습니다. 6개 모델 파일 설치를 마친 뒤 서버를 다시 시작하면 생성할 수 있습니다.",life:15000});
    }
  }
});
