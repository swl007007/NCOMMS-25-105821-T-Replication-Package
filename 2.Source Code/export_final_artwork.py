"""Render all accepted main/SI figures as editable vectors from saved results.

Run: python "2.Source Code/export_final_artwork.py"
Requires numpy, pandas, matplotlib and geopandas; no model fitting occurs here.
First run export_final_artwork_shap.py in the frozen model environment if the
Figure 2 numeric exports need rebuilding. Geometry is bundled with the artwork.
"""

import gzip
import hashlib
import json
import platform
import textwrap
from importlib.metadata import version
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib import patheffects
from matplotlib.patches import Ellipse, Patch, Polygon, Rectangle
from matplotlib.text import Text
import numpy as np
import pandas as pd

from generate_sample_imbalance_chart import apply_figure_style, build_area_frequency_distribution
from main_result_figure1_v1 import RESULTS
import generate_2022_alert_map_panel as alert_map
import generate_direct_phase3_vs_phase45_rescue_five_class_confusion as five_class

ROOT = Path(__file__).resolve().parents[1]
PG = ROOT / "2.Source Code" / "produced_graph"
RAW = ROOT / "1.Source Data"
OUT = ROOT / "writing" / "final_artwork"
DATA = OUT / "data"
MODELS = ["Forecasting", "Nowcasting", "Contemporaneous"]
PHASE_COLORS = ["#009E73", "#56B4E9", "#0072B2", "#E69F00", "#D55E00"]
POLICIES = ["direct_phase45_unweighted", "direct_phase45_sqrt_balance", "direct_phase45_full_balance"]
POLICY_NAMES = ["Unweighted", "Square-root balance", "Full balance"]
POLICY_COLORS = ["#61DDAA", "#F6BD16", "#E8684A"]
POINT, FIT, IDENTITY = "#E69F00", "#D55E00", "#777777"
AUDIT = {"sources": {}, "figures": {}}


def record(path):
    AUDIT["sources"][path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    record(path)
    return pd.read_csv(path)


def canvas(height):
    apply_figure_style()
    mpl.rcParams.update({"font.size": 6, "axes.titlesize": 7, "axes.labelsize": 6,
        "xtick.labelsize": 5, "ytick.labelsize": 5, "legend.fontsize": 6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.bbox": None, "figure.dpi": 150, "mathtext.fontset": "dejavusans"})
    return plt.figure(figsize=(180 / 25.4, height / 25.4), facecolor="white")


def title(ax, letter, text):
    ax.set_title(text, loc="left", pad=6)
    ax.text(-.13, 1.04, letter, transform=ax.transAxes, weight="bold", size=7, va="bottom")


def scatter(ax, predicted, actual, r2, fit=None):
    assert len(predicted) == len(actual) and np.isfinite([predicted, actual]).all()
    ax.scatter(predicted, actual, s=5, color=POINT, alpha=.55, edgecolors="none")
    slope, intercept = np.polyfit(predicted, actual, 1) if fit is None else fit
    line = np.array([np.min(predicted), np.max(predicted)])
    ax.plot(line, intercept + slope * line, color=FIT, lw=.8)
    ax.plot([-.03, 1.03], [-.03, 1.03], "--", color=IDENTITY, lw=.6)
    ax.set(xlim=(-.03, 1.03), ylim=(-.03, 1.03), aspect="equal")
    ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    ax.text(.04, .95, f"R² = {r2:.3f}", transform=ax.transAxes, va="top", size=6,
            bbox={"facecolor":"white", "edgecolor":"none", "alpha":.85, "pad":1})


def scatter_key(fig, y):
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=POINT, markersize=3,
                    label="Observation"), Line2D([], [], color=FIT, lw=.8, label="Linear fit"),
                    Line2D([], [], color=IDENTITY, ls="--", lw=.6, label="Equality (y = x)")],
               loc="lower center", bbox_to_anchor=(.5, y), ncol=3)


def save(fig, stem, checks):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    outside = []
    sizes = []
    for item in fig.findobj(Text):
        if not item.get_visible() or not item.get_text().strip():
            continue
        box = item.get_window_extent(renderer)
        if box.width == 0 or box.height == 0:
            continue
        sizes.append(item.get_fontsize())
        if box.x0 < -1 or box.y0 < -1 or box.x1 > bounds.width + 1 or box.y1 > bounds.height + 1:
            outside.append(item.get_text())
    assert not outside, (stem, "text outside page", outside)
    assert min(sizes) >= 5 and max(sizes) <= 7, (stem, min(sizes), max(sizes))
    assert all(not artist.get_rasterized() for artist in fig.findobj()), stem
    for ext in ("pdf", "svg", "png"):
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else {"Date": None} if ext == "svg" else None
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, facecolor="white", metadata=metadata)
    AUDIT["figures"][stem] = {"size_mm": (fig.get_size_inches() * 25.4).round(3).tolist(),
        "font_size_pt": [min(sizes), max(sizes)], "out_of_page_text": outside,
        "checks": checks, "pdf_bytes": (OUT / f"{stem}.pdf").stat().st_size}
    plt.close(fig)
    print(f"Exported {stem}", flush=True)


def figure1():
    fig = canvas(137)
    metric = {m: dict(RESULTS[m]) for m in MODELS[:2]}
    contemporary = read(PG / "all_prediction_contemporaneous_random_cv_predictions.csv")
    assert len(contemporary) == 5575
    y, pred = contemporary.overall_phase, contemporary.contemporaneous_predict
    tp = ((y >= 3) & (pred >= 3)).sum()
    actual, predicted = contemporary.phase3_actual, contemporary.phase3_contemporaneous
    metric[MODELS[2]] = {"overall_accuracy": (y == pred).mean(),
        "phase3plus_recall": tp / (y >= 3).sum(), "phase3plus_precision": tp / (pred >= 3).sum(),
        "phase3plus_r2": 1 - np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2)}
    assert np.allclose([metric[MODELS[2]][k] for k in ("overall_accuracy", "phase3plus_recall",
        "phase3plus_precision", "phase3plus_r2")], [.6934529148, .9043707832, .7972588570, .6416189986])
    ax = fig.add_axes([.09, .59, .35, .31])
    bars = ax.bar(MODELS, [metric[m]["overall_accuracy"] for m in MODELS], color="#808080", width=.65)
    ax.set(ylim=(0, .8), ylabel="Classification accuracy")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+.015,
                f"{bar.get_height():.3f}", ha="center", size=6)
    title(ax, "a", "Overall IPC phase")
    ax = fig.add_axes([.61, .59, .35, .31])
    for m, color, marker in zip(MODELS, ["#0072B2", "#E69F00", "#009E73"], ["o", "s", "^"]):
        ax.scatter(metric[m]["phase3plus_recall"], metric[m]["phase3plus_precision"],
                   s=32, c=color, marker=marker, label=m, edgecolors="white", linewidths=.3)
    ax.set(xlabel="Sensitivity (recall)", ylabel="Precision", xlim=(.895, .95), ylim=(.765, .805))
    ax.legend(loc="lower left", fontsize=5)
    title(ax, "b", "IPC Phase 3 or higher")
    for i, m in enumerate(MODELS):
        ax = fig.add_axes([.08 + i * .31, .12, .26, .30])
        if i < 2:
            frame = read(RAW / f"r2_frame_{m.lower()}.csv")
            assert frame.shape == (1170, 2)
            predicted, actual = frame.iloc[:, 0], frame.iloc[:, 1]
        else:
            predicted, actual = contemporary.phase3_contemporaneous, contemporary.phase3_actual
        r2 = 1 - np.sum((actual-predicted)**2) / np.sum((actual-actual.mean())**2)
        assert abs(r2 - metric[m]["phase3plus_r2"]) < 2e-8
        scatter(ax, predicted, actual, metric[m]["phase3plus_r2"])
        ax.set_title(m, loc="left")
        ax.set_xlabel("Predicted population share")
        if i == 0: ax.set_ylabel("Observed population share")
    fig.text(.035, .47, "c", size=7, weight="bold")
    fig.text(.08, .47, "Population share in IPC Phase 3 or higher", size=7)
    scatter_key(fig, .015)
    save(fig, "Figure_1", {"temporal_rows_per_model": 1170, "contemporaneous_rows": 5575,
                          "metrics": metric})


def figure2():
    fig = canvas(185)
    fig.text(.025, .976, "a  Forecasting", size=7, weight="bold")
    fig.text(.525, .976, "b  Nowcasting, layer 2", size=7, weight="bold")
    for col, model in enumerate(("forecasting", "nowcasting")):
        top = read(DATA / f"shap_{model}_top10.csv")
        grouped = read(DATA / f"shap_{model}_groups.csv").set_index("phase")
        signed = read(DATA / f"shap_{model}_signed_sums.csv").set_index("phase")
        assert np.allclose(grouped.sum(axis=1), 1)
        for row, phase in enumerate((3, 4)):
            selected = top.loc[top.phase == phase].sort_values("rank")
            assert list(selected.feature) == list(signed.loc[phase].abs().nlargest(10).index)
            ax = fig.add_axes([.245 + col*.5, .69-row*.30, .225, .22])
            labels = [textwrap.fill(s.replace("Enhanced Vegetation Index", "EVI")
                        .replace("Agricultural Ecological Zone", "Agroecological zone")
                        .replace("S.D.", "SD").replace("Soil P.H.", "Soil pH")
                        .replace("infrastructure", "Infrastructure"), 29) for s in selected.label]
            ax.barh(np.arange(10), selected.absolute_signed_sum, color=PHASE_COLORS[0 if phase==3 else 2])
            ax.set_yticks(np.arange(10), labels); ax.invert_yaxis()
            ax.tick_params(axis="y", length=0, pad=3)
            ax.set_title(f"Phase {phase}+", loc="left", pad=5)
            ax.set_xlabel("Absolute sum of SHAP values", size=5)
            ax.set_xlim(0, selected.absolute_signed_sum.max()*1.08)
        ax = fig.add_axes([.18 + col*.5, .12, .29, .165])
        names = [dict(Econ="Economic", Agri="Agriculture", Geo="Geography").get(n,n) for n in grouped.columns]
        y = np.arange(len(names))
        for i, color in enumerate([PHASE_COLORS[0], PHASE_COLORS[2]]):
            ax.barh(y + (i-.5)*.36, grouped.iloc[i], height=.34, color=color)
        ax.set_yticks(y, names); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
        ax.set(xlim=(0, .8), xlabel="Normalized group importance")
        fig.text(.025+col*.5, .315, ("c  Forecasting groups" if col==0 else "d  Nowcasting layer-2 groups"), size=7, weight="bold")
    fig.legend(handles=[Patch(facecolor=PHASE_COLORS[0], label="Phase 3 or higher"),
                        Patch(facecolor=PHASE_COLORS[2], label="Phase 4 or higher")],
               loc="lower center", bbox_to_anchor=(.5, .035), ncol=2)
    fig.text(.5, .015, "Feature suffixes: + = 12-month lag; * = 12-month summary (mean, or SD where labelled)",
             ha="center", size=5)
    save(fig, "Figure_2", {"training_rows_per_phase": 4405,
        "aggregation": "absolute signed sum; original notebook grouping and feature order retained"})


def arrow(ax, a, b, label=None):
    ax.annotate("", b, a, arrowprops={"arrowstyle": "-|>", "lw": .7, "color": "#333333"})
    if label:
        vertical = a[0] == b[0]
        ax.text((a[0]+b[0])/2 + (.015 if vertical else 0), (a[1]+b[1])/2+.018,
                label, ha="left" if vertical else "center", size=5)


def node(ax, x, y, text, color, shape="box", width=.17, height=.10):
    if shape == "ellipse":
        patch = Ellipse((x,y), width, height, facecolor=color, edgecolor="#555555", lw=.6)
    elif shape == "diamond":
        patch = Polygon([(x-width/2,y),(x,y+height/2),(x+width/2,y),(x,y-height/2)],
                        facecolor=color, edgecolor="#555555", lw=.6)
    else:
        patch = Rectangle((x-width/2,y-height/2),width,height,facecolor=color,edgecolor="#555555",lw=.6)
    ax.add_patch(patch); ax.text(x,y,text,ha="center",va="center",size=6)


def figure3():
    fig = canvas(100); ax = fig.add_axes([.02,.12,.96,.83]); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
    for x, phase in zip([.105,.325,.545,.765], [5,4,3,2]):
        label = "5" if phase == 5 else f"{phase}+"
        node(ax,x,.88,f"XGBoost regressor\nPhase {label}","#D6EAF8",height=.14)
        node(ax,x,.65,f"Predicted population\nshare in Phase {label}","#D5F5E3","ellipse",height=.15)
        node(ax,x,.39,"≥20%?","#E0E0E0","diamond",width=.13,height=.19)
        node(ax,x,.13,f"Phase {phase}","#AED6F1")
        arrow(ax,(x,.81),(x,.73)); arrow(ax,(x,.575),(x,.49)); arrow(ax,(x,.295),(x,.18),"Yes")
        if phase > 2: arrow(ax,(x+.065,.39),(x+.22-.065,.39),"No")
    node(ax,.945,.39,"Phase 1","#AED6F1",width=.10)
    arrow(ax,(.83,.39),(.895,.39),"No")
    ax.text(.43,.02,"Sequential IPC classification: use the highest qualifying phase",ha="center",size=6)
    fig.legend(handles=[Patch(facecolor="#D6EAF8",edgecolor="#555555",label="Regressor"),
        Patch(facecolor="#D5F5E3",edgecolor="#555555",label="Predicted share"),
        Patch(facecolor="#E0E0E0",edgecolor="#555555",label="Threshold decision"),
        Patch(facecolor="#AED6F1",edgecolor="#555555",label="Assigned phase")],
        loc="lower center",bbox_to_anchor=(.5,.02),ncol=4)
    save(fig,"Figure_3",{"rule":"highest cumulative phase with share >= 0.20; otherwise Phase 1"})


def figure4():
    raw = read(RAW / "Nowcasting_Analysis_010825.csv")
    counts = raw.overall_phase.value_counts().reindex(range(1,6),fill_value=0)
    assert counts.tolist() == [190,1968,2920,483,14]
    years = pd.crosstab(raw.date.str[:4],raw.overall_phase).reindex(columns=range(1,6),fill_value=0)
    percent = years.div(years.sum(axis=1),axis=0)*100
    fig = canvas(88)
    ax = fig.add_axes([.09,.24,.34,.63]); ax.bar(range(1,6),counts,color=PHASE_COLORS)
    ax.set(xticks=range(1,6),xlabel="IPC AFI phase",ylabel="Number of observations",ylim=(0,3300))
    for p,n in counts.items(): ax.text(p,n+60,f"{n:,}",ha="center",size=6)
    title(ax,"a","All observations, 2017–2022")
    ax = fig.add_axes([.57,.24,.39,.63]); bottom=np.zeros(len(years))
    for i,color in enumerate(PHASE_COLORS):
        ax.bar(years.index,percent.iloc[:,i],bottom=bottom,color=color,width=.7); bottom+=percent.iloc[:,i]
    ax.set(ylim=(0,100),ylabel="Share of observations (%)",xlabel="Year")
    title(ax,"b","Annual phase distribution")
    labels = ["1: Minimal","2: Stressed","3: Crisis","4: Emergency","5: Famine"]
    fig.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(PHASE_COLORS,labels)],
               loc="lower center",bbox_to_anchor=(.5,.035),ncol=5,fontsize=5)
    save(fig,"Figure_4",{"phase_counts":counts.to_dict(),"n":5575})


def supplementary1():
    raw = read(RAW / "Forecasting_Analysis_010825.csv")
    distribution = build_area_frequency_distribution(raw)
    counts = distribution.groupby("observation_frequency").area_count.sum().reindex(range(1,21),fill_value=0)
    assert counts.sum()==1198 and sum(counts.index*counts)==5575 and counts.loc[1:2].sum()==510
    fig=canvas(90); ax=fig.add_axes([.10,.18,.85,.70])
    ax.bar(counts.index,counts,color="#4C78A8",width=.82,label="Analysis areas")
    ax.set(xlabel="Observations per analysis area",ylabel="Number of analysis areas",xticks=range(1,21))
    ax.set_title("Observation frequency across 1,198 analysis areas",loc="left")
    ax.text(.97,.77,"5,575 observations\n510 areas (42.6%) observed at most twice\nMedian: 3 observations; maximum: 20",transform=ax.transAxes,ha="right",va="top",size=6)
    ax.legend(loc="upper right")
    save(fig,"Supplementary_Figure_1",{"areas":1198,"observations":5575,"areas_at_most_twice":510})


def supplementary2():
    records, thresholds = alert_map.add_top30_flags(alert_map.prepare_area_records(
        read(RAW / "All_prediction.csv"),read(RAW / "Forecasting_Analysis_010825.csv")))
    assert len(records)==646 and records.top30_forecast.sum()==207 and records.top30_actual.sum()==195
    shapes=[]
    for name in ("analysis_areas","natural_earth_countries"):
        path=DATA / f"{name}.geojson.gz"; record(path)
        with gzip.open(path,"rt") as stream: shapes.append(gpd.GeoDataFrame.from_features(json.load(stream)["features"],crs=4326).to_crs(3857))
    areas=shapes[0].merge(records,on="area_id",validate="one_to_one")
    assert len(areas)==646
    fig=canvas(165)
    xmin,ymin,xmax,ymax=areas.total_bounds
    for i,(field,label,cutoff) in enumerate([("top30_forecast","Forecasting",thresholds["forecasting"]),
                                           ("top30_actual","Observed",thresholds["actual"]) ]):
        ax=fig.add_axes([.025,.54-i*.47,.95,.40]); ax.set_facecolor("white")
        shapes[1].plot(ax=ax,color="#EEEEEE",edgecolor="#BBBBBB",linewidth=.25)
        areas.plot(ax=ax,color=np.where(areas[field]==1,"#D55E00","#0072B2"),edgecolor="white",linewidth=.08)
        ax.set(xlim=(xmin-2e5,xmax+2e5),ylim=(ymin-2e5,ymax+2e5),xticks=[],yticks=[]); ax.axis("off")
        ax.set_title(f"{'a' if i==0 else 'b'}  {label}: 70th-percentile cutoff = {cutoff:.2f}",loc="left",pad=4)
    fig.legend(handles=[Patch(facecolor="#D55E00",label="At or above 70th percentile"),
                        Patch(facecolor="#0072B2",label="Other evaluated areas"),
                        Patch(facecolor="#EEEEEE",edgecolor="#BBBBBB",label="Background land"),
                        Line2D([],[],color="#BBBBBB",lw=.6,label="Country boundary"),
                        Line2D([],[],color="white",lw=.6,label="Analysis-area boundary",
                               path_effects=[patheffects.Stroke(linewidth=2,foreground="#AAAAAA"),patheffects.Normal()]),
                        Patch(facecolor="white",edgecolor="#BBBBBB",label="Water")],
               loc="lower center",bbox_to_anchor=(.5,.005),ncol=3,fontsize=5)
    save(fig,"Supplementary_Figure_2",{"areas":646,"forecast_alerts":207,"observed_alerts":195,"thresholds":thresholds})


def supplementary3():
    source, primary=five_class.load_and_validate_source()
    for path in (five_class.DEFAULT_INPUT_CSV,five_class.DEFAULT_CONFIGURATION,
                 five_class.DEFAULT_CONTEMPORANEOUS_INPUT_CSV,five_class.DEFAULT_CONTEMPORANEOUS_CONFIGURATION,
                 five_class.DEFAULT_CONTEMPORANEOUS_PREDICTIONS_CSV): record(path)
    fig=canvas(180)
    for row,model in enumerate(MODELS):
        for col,method in enumerate(["frozen_base",primary[model]]):
            ax=fig.add_axes([.20+col*.37,.69-row*.28,.28,.22])
            cells=source.loc[(source.task==model)&(source.method==method)].set_index(["actual_phase","predicted_phase"])
            assert len(cells)==25 and cells["count"].sum()==(5575 if row==2 else 1170)
            groups=[(1,),(2,),(3,),(4,),(5,)] if col==0 else [(1,),(2,),(3,),(4,5)]
            for a in range(1,6):
                for p,group in enumerate(groups):
                    subset=cells.loc[[(a,j) for j in group]]; share=subset.actual_row_share.sum()
                    color=mpl.colormaps["cividis"](share)
                    ax.add_patch(Rectangle((p,a-1),1,1,fc=color,ec="white",lw=.5))
                    ax.text(p+.5,a-.5,f"{int(subset['count'].sum())}\n{share:.1%}",ha="center",va="center",size=5,
                            color="white" if share<.45 else "black")
            ax.set(xlim=(0,len(groups)),ylim=(5,0),aspect="equal")
            ax.set_xticks(np.arange(len(groups))+.5,[f"P{p}" for p in range(1,6)] if col==0 else ["P1","P2","P3","P4/5"])
            ax.set_yticks(np.arange(5)+.5,[f"P{p}" for p in range(1,6)])
            ax.tick_params(length=0)
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.set_title(f"{chr(97+row*2+col)}  "+("Frozen model" if col==0 else POLICY_NAMES[POLICIES.index(method)]),loc="left")
            if row==2: ax.set_xlabel("Predicted IPC phase")
        fig.text(.02,.79-row*.28,model+f"\nn = {5575 if row==2 else 1170:,}",va="center",size=6)
    cax=fig.add_axes([.90,.13,.018,.76]); cb=fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0,1),cmap="cividis"),cax=cax)
    cb.solids.set_rasterized(False); cb.set_label("Share within actual phase",size=6); cb.ax.tick_params(labelsize=5)
    fig.text(.195,.955,"Before rescue",size=7,weight="bold"); fig.text(.565,.955,"After rescue",size=7,weight="bold")
    fig.text(.14,.52,"Actual IPC phase",rotation=90,va="center",size=6)
    fig.text(.5,.025,"Cells: count and within-actual-phase percentage. P = IPC phase; P4/5 = combined phases 4 and 5.",ha="center",size=5)
    save(fig,"Supplementary_Figure_3",{"cells":300,"primary_policies":primary,"source_manifest_hash_checks":"passed"})


def supplementary4():
    data=read(PG/"phase_cumulative_three_model_scatter_predictions.csv")
    metrics=read(PG/"phase_cumulative_three_model_scatter_metrics.csv")
    assert len(data)==23745 and len(metrics)==9
    fig=canvas(180)
    for row,phase in enumerate([2,3,4]):
        for col,model in enumerate(MODELS):
            d=data.loc[(data.model==model)&(data.phase==phase)]
            m=metrics.loc[(metrics.model==model)&(metrics.phase==phase)].iloc[0]
            assert len(d)==m.n_test==(5575 if col==2 else 1170)
            r2=1-np.sum((d.actual-d.predicted)**2)/np.sum((d.actual-d.actual.mean())**2)
            assert abs(r2-m.r2)<1e-10
            ax=fig.add_axes([.10+col*.30,.70-row*.275,.24,.205])
            scatter(ax,d.predicted,d.actual,m.r2,(m.slope,m.intercept))
            row_values=data.loc[data.phase==phase,["actual","predicted"]].to_numpy()
            upper=row_values.max(); padding=max(.01,upper*.04)
            ax.set(xlim=(-padding,upper+padding),ylim=(-padding,upper+padding))
            if phase==4:
                ax.set_xticks([0,.2,.4,.6]); ax.set_yticks([0,.2,.4,.6])
            ax.text(.04,.84,f"Intercept = {m.intercept:.3f}\nSlope = {m.slope:.3f}",transform=ax.transAxes,va="top",size=5,
                    bbox={"facecolor":"white","edgecolor":"none","alpha":.85,"pad":1})
            ax.set_title(f"{chr(97+row*3+col)}  Phase {phase}+",loc="left")
            if row==2: ax.set_xlabel("Predicted population share")
            if col==0: ax.set_ylabel("Observed population share")
    for col,model in enumerate(MODELS): fig.text(.22+col*.30,.963,model,ha="center",size=7,weight="bold")
    scatter_key(fig,.025)
    save(fig,"Supplementary_Figure_4",{"points":23745,"panels":9,"saved_r2_recomputed":"passed"})


def supplementary5():
    binary=[]; gates=[]
    for dirname in ("direct_phase3_vs_phase45_rescue","direct_phase3_vs_phase45_rescue_contemporaneous"):
        binary.append(read(PG/dirname/(dirname+"_binary_confusion_matrices.csv")))
        gates.append(read(PG/dirname/(dirname+"_gate_pooled_metrics.csv")))
    binary=pd.concat(binary); gates=pd.concat(gates)
    fig=canvas(155)
    for row,model in enumerate(MODELS):
        primary=gates.loc[(gates.task==model)&gates.primary_selected,"method"].tolist()
        assert len(primary)==1
        for col,(method,color) in enumerate(zip(POLICIES,POLICY_COLORS)):
            d=binary.loc[(binary.task==model)&(binary.method==method)&
                         (binary.population_scope==("full_oof" if row==2 else "benchmark"))]
            assert len(d)==4 and d["count"].sum()==(5575 if row==2 else 1170)
            ax=fig.add_axes([.24+col*.25,.72-row*.25,.18,.20])
            for cell in d.itertuples():
                share=cell.actual_row_share; fill=1-(1-np.array(to_rgb(color)))*(.12+.78*share)
                ax.add_patch(Rectangle((cell.predicted_binary,cell.actual_binary),1,1,fc=fill,ec="white",lw=.5))
                ax.text(cell.predicted_binary+.5,cell.actual_binary+.5,f"{cell.count}\n{share:.1%}",ha="center",va="center",size=6)
            ax.set(xlim=(0,2),ylim=(2,0),aspect="equal")
            ax.set_xticks([.5,1.5],["Below P4","P4/5"]); ax.set_yticks([.5,1.5],["Below P4","P4/5"])
            ax.tick_params(length=0)
            for spine in ax.spines.values(): spine.set_visible(method==primary[0]); spine.set_linewidth(1.1); spine.set_color("black")
            if row==0: ax.set_title(POLICY_NAMES[col],loc="center",size=6,pad=8)
            if row==2: ax.set_xlabel("Predicted class")
            if col==0: ax.set_ylabel("Actual class")
        fig.text(.015,.82-row*.25,model+f"\nn = {5575 if row==2 else 1170:,}",size=6,va="center")
    for col,color in enumerate(POLICY_COLORS):
        ax=fig.add_axes([.245+col*.25,.105,.17,.022])
        for i in range(100):
            fill=1-(1-np.array(to_rgb(color)))*(.12+.78*(i+.5)/100)
            ax.add_patch(Rectangle((i/100,0),.01,1,fc=fill,ec="none"))
        ax.set(xlim=(0,1),ylim=(0,1),yticks=[],xticks=[0,1],xticklabels=["0%","100%"])
        ax.tick_params(length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.text(.5,.035,"Shading: percentage within actual class. Black frame: selected primary policy.",ha="center",size=5)
    fig.text(.5,.012,"P4/5 = IPC phases 4 or 5; below P4 = phases 1–3.",ha="center",size=5)
    save(fig,"Supplementary_Figure_5",{"matrices":9,"cells":36,"temporal_rows":1170,"contemporaneous_rows":5575})


def supplementary6():
    fig=canvas(93); ax=fig.add_axes([.04,.13,.92,.80]); ax.set(xlim=(-1.5,7.2),ylim=(-.75,5.1)); ax.axis("off")
    train,valid,test="#D6EAF8","#F8DDA0","#D9D9D9"
    for col,year in enumerate(range(2017,2023)): ax.text(col+.5,4.8,str(year),ha="center",size=7)
    for row in range(4):
        y=4-row
        ax.text(-.25,y+.1,f"Fold {row+1}",ha="right",va="center",size=6)
        for col in range(row+2):
            ax.add_patch(Rectangle((col+.04,y-.23),.92,.66,fc=train if col<=row else valid,ec="white"))
        arrow(ax,(row+.75,y+.10),(row+1.2,y+.10))
    ax.plot([6.15,6.35,6.35,6.15],[.77,.77,4.43,4.43],color="#555555",lw=.7)
    ax.text(6.5,2.6,"Hyperparameter\nselection",va="center",size=6)
    ax.text(-.25,-.1,"Final fit",ha="right",va="center",size=6)
    for col in range(6): ax.add_patch(Rectangle((col+.04,-.43),.92,.66,fc=train if col<5 else test,ec="white"))
    arrow(ax,(4.75,-.1),(5.2,-.1))
    fig.legend(handles=[Patch(fc=train,label="Training"),Patch(fc=valid,label="Validation"),
                        Patch(fc=test,label="2022 holdout"),Line2D([],[],marker=">",color="#333333",lw=.7,label="Fit → evaluate")],
               loc="lower center",bbox_to_anchor=(.5,.025),ncol=4)
    save(fig,"Supplementary_Figure_6",{"tuning_validation_years":[2018,2019,2020,2021],"holdout_year":2022})


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for draw in (figure1,figure2,figure3,figure4,supplementary1,supplementary2,supplementary3,supplementary4,supplementary5,supplementary6): draw()
    assert len(AUDIT["figures"])==10
    record(Path(__file__).resolve())
    for filename in ("generate_sample_imbalance_chart.py", "main_result_figure1_v1.py",
                     "generate_2022_alert_map_panel.py", "generate_direct_phase3_vs_phase45_rescue_five_class_confusion.py"):
        record(ROOT / "2.Source Code" / filename)
    AUDIT["render_environment"]={"python":platform.python_version(),"packages":{n:version(n) for n in ["numpy","pandas","matplotlib","geopandas"]}}
    AUDIT["outputs"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.suffix in (".pdf",".svg",".png")}
    assert len(AUDIT["outputs"])==30
    (OUT/"artwork_audit.json").write_text(json.dumps(AUDIT,indent=2,default=int)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
