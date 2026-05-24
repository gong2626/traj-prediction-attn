from __future__ import annotations

import csv
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "slides"
OUT_PPTX = OUT_DIR / "traj_forecast_group_meeting_zh.pptx"

FIG1 = ROOT / "docs" / "paper" / "figures" / "fig1_method_architecture.png"
FIG2 = ROOT / "docs" / "paper" / "figures" / "fig2_main_results_overview.png"
FIG3 = ROOT / "docs" / "paper" / "figures" / "fig3_h3_refinement_tradeoff.png"

MAIN_RESULTS = ROOT / "docs" / "paper" / "tables" / "main_results.csv"
ABLATION_RESULTS = ROOT / "docs" / "paper" / "tables" / "structural_ablation.csv"


BG = RGBColor(248, 250, 252)
FG = RGBColor(30, 41, 59)
MUTED = RGBColor(71, 85, 105)
ACCENT = RGBColor(37, 99, 235)
ACCENT_2 = RGBColor(14, 116, 144)
BORDER = RGBColor(203, 213, 225)
GOOD = RGBColor(22, 163, 74)


def add_bg(slide):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = BG


def add_title(slide, title: str, subtitle: str | None = None):
    box = slide.shapes.add_textbox(Inches(0.55), Inches(0.3), Inches(11.9), Inches(0.7))
    tf = box.text_frame
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title
    run.font.name = "Noto Sans CJK SC"
    run.font.bold = True
    run.font.size = Pt(26)
    run.font.color.rgb = FG
    if subtitle:
        sub = slide.shapes.add_textbox(Inches(0.58), Inches(0.88), Inches(11.4), Inches(0.42))
        tf2 = sub.text_frame
        p2 = tf2.paragraphs[0]
        r2 = p2.add_run()
        r2.text = subtitle
        r2.font.name = "Noto Sans CJK SC"
        r2.font.size = Pt(11)
        r2.font.color.rgb = MUTED


def add_bullets(slide, items: list[str], left: float, top: float, width: float, height: float, font_size: int = 18):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for idx, item in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = item
        p.level = 0
        p.font.name = "Noto Sans CJK SC"
        p.font.size = Pt(font_size)
        p.font.color.rgb = FG
        p.space_after = Pt(8)


def add_note_box(slide, title: str, lines: list[str], left: float, top: float, width: float, height: float, fill_rgb=RGBColor(255, 255, 255)):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_rgb
    shape.line.color.rgb = BORDER
    shape.line.width = Pt(1.2)
    tf = shape.text_frame
    tf.clear()
    p0 = tf.paragraphs[0]
    r0 = p0.add_run()
    r0.text = title
    r0.font.name = "Noto Sans CJK SC"
    r0.font.bold = True
    r0.font.size = Pt(15)
    r0.font.color.rgb = ACCENT
    for line in lines:
        p = tf.add_paragraph()
        p.text = line
        p.font.name = "Noto Sans CJK SC"
        p.font.size = Pt(12)
        p.font.color.rgb = FG
        p.space_after = Pt(5)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add_table(slide, rows: list[list[str]], left: float, top: float, width: float, height: float, col_widths: list[float] | None = None):
    nrows = len(rows)
    ncols = len(rows[0])
    table = slide.shapes.add_table(nrows, ncols, Inches(left), Inches(top), Inches(width), Inches(height)).table
    if col_widths:
        for idx, w in enumerate(col_widths):
            table.columns[idx].width = Inches(w)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.text = value
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(241, 245, 249) if i == 0 else RGBColor(255, 255, 255)
            cell_text = cell.text_frame
            cell_text.word_wrap = True
            for p in cell_text.paragraphs:
                p.alignment = PP_ALIGN.CENTER if i == 0 or j > 0 else PP_ALIGN.LEFT
                for run in p.runs:
                    run.font.name = "Noto Sans CJK SC"
                    run.font.size = Pt(11 if i == 0 else 10.5)
                    run.font.bold = bool(i == 0)
                    run.font.color.rgb = FG


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank = prs.slide_layouts[6]

    # Slide 1
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_title(s, "全球航空轨迹多时间尺度区域预测", "组会汇报 | 方法原理与实验结果")
    add_note_box(
        s,
        "研究目标",
        [
            "基于大规模开放航空轨迹数据，预测未来 10/30/60/120 分钟所在 H3 区域分布。",
            "关注的不只是 top-k 命中，还包括概率质量、候选区域紧致性和长时域稳定性。",
        ],
        0.65, 1.35, 5.8, 1.55, RGBColor(239, 246, 255),
    )
    add_note_box(
        s,
        "当前定位",
        [
            "任务：层级 H3 区域概率预测，而不是传统经纬度点回归。",
            "数据：ADRR 构建的 2024 年 3 月商业航班档案，full-scale 约 2071 万窗口。",
            "方法主线：从 hierarchical GRU 演化到 multiscale horizon-query Transformer。",
        ],
        6.75, 1.35, 5.9, 1.8, RGBColor(240, 253, 250),
    )
    add_bullets(
        s,
        [
            "汇报重点 1：为什么把问题定义成层级 H3 区域预测",
            "汇报重点 2：所提出多尺度模型的核心结构",
            "汇报重点 3：与内部基线、外部 adapted baselines 的对比结果",
        ],
        0.8, 3.45, 11.7, 2.2, 20,
    )

    # Slide 2
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_title(s, "问题定义", "输入历史轨迹，输出未来多个 horizon 的层级 H3 区域概率")
    add_note_box(
        s,
        "输入",
        [
            "30 分钟历史窗口，5 分钟采样，共 6 个时间步。",
            "每步包含：H3 区域 id、父层区域 id、18 维数值特征、2 维 mask 特征。",
            "附加 9 维上下文特征。",
        ],
        0.7, 1.35, 3.8, 2.0, RGBColor(255, 255, 255),
    )
    add_note_box(
        s,
        "输出",
        [
            "预测 horizon：10 / 30 / 60 / 120 min。",
            "对每个 horizon 同时预测：coarse H3、child H3、airborne/landed 状态。",
            "细粒度目标分辨率：[3,3,2,2]。",
        ],
        4.75, 1.35, 3.8, 2.0, RGBColor(255, 255, 255),
    )
    add_note_box(
        s,
        "评价",
        [
            "分类：Top1 / Top3 / Top5",
            "概率质量：NLL / Brier / ECE",
            "候选区域：Union Cover / Union Cells / Union Area / Cover@Budget",
        ],
        8.8, 1.35, 3.8, 2.0, RGBColor(255, 255, 255),
    )
    add_bullets(
        s,
        [
            "为什么不用单点坐标误差：对于空管/区域监控场景，更关心未来落在哪些可操作区域中。",
            "为什么做层级预测：直接在巨大细粒度空间上分类太难，coarse-to-fine 更符合 H3 父子层级。",
            "为什么看候选区域：高 top-k 但候选区域过大，运行价值仍然有限。",
        ],
        0.8, 3.8, 11.8, 2.6, 18,
    )

    # Slide 3
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_title(s, "方法框架", "统一层级 H3 输出头 + horizon-aware 时间建模")
    s.shapes.add_picture(str(FIG1), Inches(0.65), Inches(1.25), width=Inches(8.1))
    add_note_box(
        s,
        "核心演化路径",
        [
            "Hierarchical GRU：所有 horizon 共用一个时间摘要，长短时域共享同一表示。",
            "Single-scale horizon-query Transformer：不同 horizon 用不同 query 从历史中读取信息。",
            "Multiscale Transformer：在此基础上增加显式粗尺度时间分支。",
        ],
        8.95, 1.35, 3.8, 2.35, RGBColor(255, 255, 255),
    )
    add_note_box(
        s,
        "关键直觉",
        [
            "10/30 分钟更依赖局部机动信息。",
            "60/120 分钟更依赖阶段性趋势和粗尺度演化。",
            "因此，用一个单尺度时间摘要往往不够。",
        ],
        8.95, 3.95, 3.8, 1.95, RGBColor(239, 246, 255),
    )

    # Slide 4
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_title(s, "所提出模型的关键设计", "多尺度 horizon-query Transformer 关注中长时域区域预测")
    add_bullets(
        s,
        [
            "细尺度分支：保留 6 步历史序列，捕捉局部机动与短期区域切换。",
            "粗尺度分支：对细尺度序列做 stride=2 的 masked pooling，再做额外 Transformer 编码。",
            "每个 horizon 有独立 query，同时从 fine/coarse 两个时间尺度读取信息，再融合成该 horizon 的解码特征。",
            "输出端统一采用 coarse-to-fine H3 头，保证和所有内部/外部 baseline 在同一输出协议下比较。",
        ],
        0.8, 1.35, 7.0, 3.6, 18,
    )
    add_note_box(
        s,
        "为什么这套设计合理",
        [
            "不直接换任务定义，只替换时间建模主干。",
            "既能和 GRU 比，也能和 adapted external baselines 比。",
            "理论上更适合长时域信息抽取，实验中确实在 120 分钟上更稳。",
        ],
        8.2, 1.6, 4.2, 2.4, RGBColor(240, 253, 250),
    )
    add_note_box(
        s,
        "推理后处理",
        [
            "默认启用 reachability filter。",
            "H3 refinement 只作为消融，不作为最终方法。",
        ],
        8.2, 4.35, 4.2, 1.35, RGBColor(255, 255, 255),
    )

    # Slide 5
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_title(s, "主结果对比", "与内部基线和外部 adapted baselines 的 full-scale 宏平均比较")
    main_rows = load_csv(MAIN_RESULTS)
    table_rows = [
        ["模型", "Top1", "Top3", "Top5", "NLL", "Brier", "ECE"],
    ]
    name_map = {
        "Markov baseline": "Markov",
        "Hierarchical GRU": "层级 GRU",
        "TrAISformer-H3": "TrAISformer-H3",
        "FlightBERT++-H3": "FlightBERT++-H3",
        "Transformer horizon decoder": "单尺度 Transformer",
        "Proposed multiscale model": "所提多尺度模型",
    }
    for row in main_rows:
        table_rows.append(
            [
                name_map[row["Model"]],
                f'{float(row["Top1"]):.4f}',
                f'{float(row["Top3"]):.4f}',
                f'{float(row["Top5"]):.4f}',
                f'{float(row["NLL"]):.4f}',
                f'{float(row["Brier"]):.4f}',
                f'{float(row["ECE"]):.4f}',
            ]
        )
    add_table(s, table_rows, 0.55, 1.3, 7.15, 3.55, [2.05, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    add_note_box(
        s,
        "读表结论",
        [
            "TrAISformer-H3 是当前最强 external baseline。",
            "所提多尺度模型取得最佳 Top1、最佳 Brier、最低候选 cell 数。",
            "FlightBERT++-H3 的 ECE 最低，但整体 top-k 不如前两者。",
        ],
        8.05, 1.45, 4.3, 2.0, RGBColor(255, 255, 255),
    )
    add_note_box(
        s,
        "汇报时建议的口径",
        [
            "不要说“全面最好”。",
            "更稳妥的说法是：在强 baseline 存在的情况下，所提方法在首选命中、概率质量和候选紧致性之间更均衡。",
        ],
        8.05, 3.75, 4.3, 1.9, RGBColor(239, 246, 255),
    )

    # Slide 6
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_title(s, "结构消融与后处理分析", "多尺度分支是有效的，H3 refinement 更像召回-校准权衡")
    ab_rows = load_csv(ABLATION_RESULTS)
    table_rows = [["模型", "Top1", "Top5", "NLL", "Brier", "ECE"]]
    for row in ab_rows:
        label = "单尺度 Transformer" if row["Model"] == "Transformer horizon decoder" else "所提多尺度模型"
        table_rows.append(
            [
                label,
                f'{float(row["Top1"]):.4f}',
                f'{float(row["Top5"]):.4f}',
                f'{float(row["NLL"]):.4f}',
                f'{float(row["Brier"]):.4f}',
                f'{float(row["ECE"]):.4f}',
            ]
        )
    add_table(s, table_rows, 0.55, 1.35, 5.5, 1.55, [2.1, 0.7, 0.7, 0.8, 0.8, 0.8])
    s.shapes.add_picture(str(FIG3), Inches(0.65), Inches(3.2), width=Inches(5.7))
    add_note_box(
        s,
        "消融结论 1：多尺度分支必要",
        [
            "相较单尺度 Transformer，所提模型在 Top1/Top5/NLL/Brier/ECE 上全部更优。",
            "因此收益不只是“把 GRU 换成 Transformer”。",
        ],
        6.45, 1.35, 5.95, 1.7, RGBColor(240, 253, 250),
    )
    add_note_box(
        s,
        "消融结论 2：H3 refinement 不适合作为最终方法",
        [
            "它能提高覆盖率，但会恶化校准并扩大候选区域。",
            "更适合被解释为运行中的召回-校准权衡工具。",
        ],
        6.45, 3.45, 5.95, 1.6, RGBColor(255, 255, 255),
    )

    # Slide 7
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_title(s, "当前结论与下一步", "主线已经清晰，接下来重点补强结果叙事和解码侧优化")
    add_bullets(
        s,
        [
            "结论 1：层级 H3 多 horizon 区域预测在大规模开放航空轨迹数据上是可行的。",
            "结论 2：多尺度 horizon-query Transformer 相较项目原始 GRU 基线有稳定提升。",
            "结论 3：强 external baseline 存在，说明任务并不容易，当前方法的优势主要体现在 Top1、Brier 和候选紧致性。",
        ],
        0.8, 1.4, 11.6, 2.15, 19,
    )
    add_note_box(
        s,
        "下一步计划",
        [
            "优先做 evaluation-only 解码 sweep，尝试用更大的 parent top-k 改善 Top5/NLL。",
            "若仍需结构改进，再考虑针对短时排序能力补强，而不是直接改更细网格。",
            "论文层面强调：最终方法不是“全指标最好”，而是“更均衡的运行型方案”。",
        ],
        0.85, 4.05, 11.45, 2.2, RGBColor(239, 246, 255),
    )

    prs.save(OUT_PPTX)
    print(OUT_PPTX)


if __name__ == "__main__":
    main()
