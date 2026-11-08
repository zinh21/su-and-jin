import glob
import re
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

st.set_page_config(page_title="어린이보호구역 교통안전", page_icon="🚸", layout="wide")


# =============================================================
# 데이터 읽기 (저장소 안의 CSV)
# =============================================================
def read_csv_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    with open(files[0], "rb") as f:
        raw = f.read()
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(StringIO(raw.decode(enc)), dtype=str)
        except Exception:
            continue
    return None


def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(),
                         errors="coerce")


@st.cache_data
def load_data():
    zone = read_csv_file("161601*.csv")
    traffic = read_csv_file("교통사고통계*.csv")
    if zone is None or traffic is None:
        return None, None, "CSV 파일을 찾지 못했습니다. 두 CSV가 main.py와 같은 폴더에 있는지 확인하세요."

    zone.columns = [str(c).strip() for c in zone.columns]
    traffic.columns = [str(c).strip() for c in traffic.columns]

    year_cols_z = [c for c in zone.columns if re.fullmatch(r"(19|20)\d{2}", c)]
    year_cols_t = [c for c in traffic.columns if re.fullmatch(r"(19|20)\d{2}", c)]
    if not year_cols_z or not year_cols_t:
        return None, None, f"연도 열을 찾지 못했습니다. 열 이름: {list(zone.columns)} / {list(traffic.columns)}"

    log = [
        {"자료": "어린이보호구역", "원본 행": len(zone), "원본 열": zone.shape[1],
         "원본 빈 값": int(zone.isna().sum().sum())},
        {"자료": "교통사고", "원본 행": len(traffic), "원본 열": traffic.shape[1],
         "원본 빈 값": int(traffic.isna().sum().sum())},
    ]

    zrow = None
    for _, r in zone.iterrows():
        if to_num(r[year_cols_z]).notna().all():
            zrow = r
            break
    if zrow is None:
        return None, None, "보호구역 CSV에서 숫자 행을 찾지 못했습니다."
    zones = to_num(zrow[year_cols_z]).values
    log[0]["사용한 행"] = 1

    t = traffic.copy()
    for c in ["시도", "시군구", "연도"]:
        if c in t.columns:
            t[c] = t[c].astype(str).str.strip()
    if "시도" in t.columns and (t["시도"] == "합계").any():
        t = t[t["시도"] == "합계"]
    if "시군구" in t.columns and (t["시군구"] == "합계").any():
        t = t[t["시군구"] == "합계"]

    def find_row(keyword):
        sub = t[t["연도"].str.contains(keyword, regex=False)]
        if sub.empty:
            return None
        return to_num(sub.iloc[0][year_cols_t]).values

    acc, death, injury = find_row("사고"), find_row("사망"), find_row("부상")
    if acc is None:
        return None, None, "교통사고 CSV의 '연도' 열에서 '사고' 행을 찾지 못했습니다."
    log[1]["사용한 행"] = sum(x is not None for x in [acc, death, injury])

    zdf = pd.DataFrame({"연도": [int(c) for c in year_cols_z], "지정 수": zones})
    tdf = pd.DataFrame({
        "연도": [int(c) for c in year_cols_t],
        "사고 건수": acc,
        "사망자": death if death is not None else np.nan,
        "부상자": injury if injury is not None else np.nan,
    })
    df = zdf.merge(tdf, on="연도").sort_values("연도").reset_index(drop=True)

    use_cols = ["지정 수", "사고 건수"]
    before = len(df)
    df = df.dropna(subset=use_cols).reset_index(drop=True)
    for i in (0, 1):
        log[i]["연도 행(처리 전)"] = before
        log[i]["삭제한 연도 행"] = before - len(df)
        log[i]["연도 행(처리 후)"] = len(df)

    return df, pd.DataFrame(log), None


def pct(a, b):
    return (a - b) / b * 100 if b else 0


# =============================================================
# 회귀 평가 (학습용 vs 시험용)
# =============================================================
def evaluate(df, target, features, split_year, max_depth, n_trees):
    train, test = df[df["연도"] < split_year], df[df["연도"] >= split_year]
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    models = {
        "베이스라인(평균)": DummyRegressor(strategy="mean"),
        "선형 회귀": LinearRegression(),
        "결정트리": DecisionTreeRegressor(max_depth=max_depth, random_state=42),
        "랜덤 포레스트": RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth,
                                            random_state=42),
    }
    X_all, y_all = df[features], df[target]
    rows, preds = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        p_test = model.predict(X_test)
        p_train = model.predict(X_train)
        preds[name] = p_test
        loo = cross_val_predict(model, X_all, y_all, cv=LeaveOneOut())
        rows.append({
            "모델": name,
            "학습 MAE": mean_absolute_error(y_train, p_train),
            "시험 MAE": mean_absolute_error(y_test, p_test),
            "시험 RMSE": np.sqrt(mean_squared_error(y_test, p_test)),
            "학습 R²": r2_score(y_train, p_train),
            "시험 R²": r2_score(y_test, p_test) if len(y_test) > 1 else np.nan,
            "LOOCV MAE": mean_absolute_error(y_all, loo),
            "LOOCV R²": r2_score(y_all, loo),
        })
    result = pd.DataFrame(rows)
    return result, preds, models, train, test


def to_markdown_table(result):
    cols = [c for c in result.columns if c != "모델"]
    head = "| " + " | ".join(result.columns) + " |"
    div = "|" + "|".join(["---"] * len(result.columns)) + "|"
    body = ["| " + " | ".join([r["모델"]] + [f"{r[c]:.3f}" for c in cols]) + " |"
            for _, r in result.iterrows()]
    return "\n".join([head, div] + body)


# =============================================================
# 분류 도우미
# =============================================================
def safe_div(a, b):
    return a / b if b else 0.0


def cm_counts(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def cls_metrics(y_true, pred):
    return {
        "정확도": accuracy_score(y_true, pred),
        "정밀도": precision_score(y_true, pred, zero_division=0),
        "재현율": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
    }


def cm_figure(tn, fp, fn, tp, title):
    cm = np.array([[tn, fp], [fn, tp]])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    x=["예측: 적음(0)", "예측: 많음(1)"],
                    y=["실제: 적음(0)", "실제: 많음(1)"], title=title)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_traces(textfont_size=28)
    return fig


# =============================================================
# 페이지 1: 교통안전 분석
# =============================================================
def page_analysis(df, log):
    st.title("🚸 어린이보호구역 교통안전 분석")
    st.markdown("어린이보호구역 지정 수와 교통사고 변화를 살펴봅니다.")

    st.header("🗂️ 데이터 개요")
    c1, c2, c3 = st.columns(3)
    c1.metric("정리된 표 행 개수", f"{len(df):,}")
    c2.metric("열 개수", f"{df.shape[1]:,}")
    c3.metric("빈 값 총합", f"{int(df.isna().sum().sum()):,}")
    ov = pd.DataFrame({"열 이름": df.columns, "자료형": df.dtypes.astype(str).values,
                       "빈 값 개수": df.isna().sum().values})
    st.dataframe(ov, hide_index=True, use_container_width=True)
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.header("🧹 결측치·이상치 처리 결과")
    st.dataframe(log, hide_index=True, use_container_width=True)
    st.caption(
        "결측치: 분석에 쓰는 열(지정 수, 사고 건수)에 빈 값이 있는 연도 행은 삭제합니다. "
        "삭제한 연도 행이 0이면 빈 값이 없었다는 뜻입니다. "
        "이상치: 연도가 16개뿐이라 IQR 같은 방법이 의미가 없어 처리하지 않았습니다."
    )

    st.header("📊 그래프와 발견한 점")
    units = {"지정 수": "개", "사고 건수": "건", "사망자": "명", "부상자": "명"}
    colors = {"지정 수": "#00CC96", "사고 건수": "#EF553B", "사망자": "#636EFA", "부상자": "#FFA15A"}
    cols_list = [c for c in ["지정 수", "사고 건수", "사망자", "부상자"] if df[c].notna().any()]

    for i in range(0, len(cols_list), 2):
        row = st.columns(2)
        for j, col in enumerate(cols_list[i:i + 2]):
            with row[j]:
                fig = px.bar(df, x="연도", y=col, text=col, title=f"연도별 {col}",
                             color_discrete_sequence=[colors[col]])
                fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
                fig.update_layout(xaxis=dict(type="category"), yaxis_title=None)
                st.plotly_chart(fig, use_container_width=True)

                s, y = df[col], df["연도"]
                d = s.iloc[-1] - s.iloc[0]
                word = "증가" if d > 0 else "감소" if d < 0 else "변화 없음"
                u = units[col]
                st.info(
                    f"🔎 **발견:** {col}은(는) {y.iloc[0]}년 대비 {y.iloc[-1]}년에 **{word}**했고 "
                    f"최댓값은 {y[s.idxmax()]}년입니다.  \n"
                    f"📎 **근거:** {y.iloc[0]}년 {s.iloc[0]:,.0f}{u} → {y.iloc[-1]}년 {s.iloc[-1]:,.0f}{u} "
                    f"({d:+,.0f}{u}, {pct(s.iloc[-1], s.iloc[0]):+.1f}%), "
                    f"최대 {s.max():,.0f}{u}, 최소 {s.min():,.0f}{u}"
                )

    st.subheader("📈 시작 연도=100 변화율 비교")
    idx = pd.DataFrame({"연도": df["연도"]})
    for c in cols_list:
        idx[c] = df[c] / df[c].iloc[0] * 100
    fig = px.line(idx.melt(id_vars="연도", var_name="항목", value_name="지수"),
                  x="연도", y="지수", color="항목", markers=True)
    fig.update_layout(xaxis=dict(type="category"))
    st.plotly_chart(fig, use_container_width=True)
    last = idx.iloc[-1].drop("연도").astype(float)
    st.info(f"🔎 **발견:** {df['연도'].iloc[-1]}년 기준 가장 높은 항목은 **{last.idxmax()}**, "
            f"가장 낮은 항목은 **{last.idxmin()}**입니다.  \n"
            f"📎 **근거:** " + ", ".join(f"{k} {v:.1f}" for k, v in last.items()))

    st.warning("지정 수 변화가 사고 변화의 원인이라고 단정할 수 없습니다. "
               "통행량, 안전시설 등 다른 자료가 필요합니다.")


# =============================================================
# 페이지 2: 회귀 모델 비교
# =============================================================
def page_regression(df):
    st.title("📐 지정 수로 사고 건수 예측 (회귀 모델 비교)")

    min_y, max_y = int(df["연도"].min()), int(df["연도"].max())

    st.sidebar.header("⚙️ 회귀 설정")
    targets = [c for c in ["사고 건수", "사망자", "부상자"] if df[c].notna().any()]
    target = st.sidebar.selectbox("예측할 값(y)", targets)
    use_year = st.sidebar.checkbox("연도도 특성에 포함", value=False)
    split_year = st.sidebar.slider(
        "테스트 시작 연도 (이 연도 전 = 학습, 이 연도 이후 = 테스트)",
        min_y + 8, max_y - 1, min(2021, max_y - 1),
    )
    max_depth = st.sidebar.slider("트리 max_depth", 1, 6, 2)
    n_trees = st.sidebar.slider("랜덤 포레스트 나무 수", 20, 300, 100, 20)

    features = ["지정 수"] + (["연도"] if use_year else [])
    result, preds, models, train, test = evaluate(df, target, features, split_year,
                                                  max_depth, n_trees)
    st.caption(f"학습 {len(train)}개({train['연도'].min()}~{train['연도'].max()}) / "
               f"테스트 {len(test)}개({test['연도'].min()}~{test['연도'].max()}) - 시간 순서 분할")

    cols = [c for c in result.columns if c != "모델"]
    st.header("📋 성능 비교표 (학습용 vs 시험용)")
    st.dataframe(result.style.format({c: "{:.3f}" for c in cols}),
                 hide_index=True, use_container_width=True)
    st.caption("학습 지표만 좋고 시험 지표가 나쁘면 학습 데이터를 외운 과적합 신호입니다. "
               "모델 비교는 시험 MAE·RMSE·R²와 LOOCV로 하세요. R²가 0보다 작으면 평균 예측보다 못합니다.")

    st.header("📈 전체 기간 실제값과 테스트 기간 예측")
    full = pd.DataFrame({"연도": df["연도"], "실제": df[target]})
    pred_df = pd.DataFrame({"연도": test["연도"].values})
    for name, p in preds.items():
        pred_df[name] = p
    merged_plot = full.merge(pred_df, on="연도", how="left")
    long = merged_plot.melt(id_vars="연도", var_name="구분", value_name=target)
    fig = px.line(long.dropna(), x="연도", y=target, color="구분", markers=True)
    fig.add_vrect(x0=min_y - 0.5, x1=split_year - 0.5, fillcolor="gray", opacity=0.15,
                  annotation_text=f"학습 구간 ({min_y}~{split_year - 1})",
                  annotation_position="top left")
    fig.update_layout(xaxis=dict(type="linear", dtick=1))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"회색 영역이 학습 구간이에요. 학습은 항상 {min_y}년부터 시작하고, "
               f"예측선은 테스트 구간({split_year}~{max_y})에만 나타나요.")

    real = result[result["모델"] != "베이스라인(평균)"]
    best = real.loc[real["시험 MAE"].idxmin()]
    base = result[result["모델"] == "베이스라인(평균)"].iloc[0]
    st.info(f"🔎 **발견:** 시험 MAE가 가장 낮은 모델은 **{best['모델']}**입니다.  \n"
            f"📎 **근거:** 시험 MAE {best['시험 MAE']:.2f} (베이스라인 {base['시험 MAE']:.2f}), "
            f"시험 R² {best['시험 R²']:.3f}, LOOCV R² {best['LOOCV R²']:.3f}")

    st.header("🔍 선형 회귀 해석")
    lin = models["선형 회귀"]
    for f, c in zip(features, lin.coef_):
        st.write(f"`{f}` 1 증가 시 {target} 변화: **{c:+.4f}**")
    st.write(f"절편: {lin.intercept_:.2f}")

    st.header("📝 README에 붙여넣을 표")
    md = to_markdown_table(result)
    st.code(md, language="markdown")
    st.download_button("표를 .md로 저장", md, file_name="regression_comparison.md")

    st.warning("표본이 16개뿐이고 지정 수와 사고 건수는 둘 다 시간에 따라 변해 상관이 있어도 "
               "원인이라 할 수 없습니다. 트리 계열은 학습 범위 밖 값을 예측하지 못합니다.")


# =============================================================
# 페이지 3: 분류 모델 비교 (사고 '많은 해' 분류)
# =============================================================
def page_classification(df):
    st.title("🎯 분류: 사고가 '많은 해'일까?")
    st.markdown("사고 건수가 기준보다 많은 해를 **1(많음)**, 아니면 **0(적음)** 으로 두고 분류합니다.")

    c1, c2, c3, c4 = st.columns(4)
    q = c1.slider("'많은 해' 기준 분위수", 0.5, 0.8, 0.7, 0.05,
                  help="0.7이면 사고 건수 상위 30%인 해를 '많음'으로 둡니다.")
    use_year = c2.checkbox("연도도 특성에 포함", value=False)
    max_depth = c3.slider("트리 max_depth", 1, 6, 2)
    n_trees = c4.slider("랜덤 포레스트 나무 수", 20, 300, 100, 20)

    features = ["지정 수"] + (["연도"] if use_year else [])
    cut = float(df["사고 건수"].quantile(q))
    y = (df["사고 건수"] >= cut).astype(int).values
    X = df[features]

    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    st.write(f"사고 건수 **{cut:,.1f}건 이상**이면 '많음' → 많음 {n_pos}개 / 적음 {n_neg}개 "
             f"(많음 연도: {', '.join(map(str, df.loc[y == 1, '연도']))})")
    if min(n_pos, n_neg) < 2:
        st.error("한쪽 클래스가 2개 미만이라 평가할 수 없어요. 기준 분위수를 조정하세요.")
        return
    st.caption("자료가 연도별 16행뿐이라 시험 데이터를 따로 떼지 않고, 한 해씩 빼고 나머지로 학습해 "
               "그 해를 예측하는 방식(LOOCV)으로 16개 전체의 예측을 모았습니다.")

    models = {
        "로지스틱 회귀": make_pipeline(StandardScaler(),
                                  LogisticRegression(class_weight="balanced", max_iter=1000)),
        "결정트리": DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced",
                                       random_state=42),
        "랜덤 포레스트": RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth,
                                            class_weight="balanced", random_state=42),
    }
    logit_off = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    loo = LeaveOneOut()

    probas = {n: cross_val_predict(m, X, y, cv=loo, method="predict_proba")[:, 1]
              for n, m in models.items()}
    proba_off = cross_val_predict(logit_off, X, y, cv=loo, method="predict_proba")[:, 1]
    base_pred = cross_val_predict(DummyClassifier(strategy="most_frequent"), X, y, cv=loo)

    st.header("1️⃣ 가중치를 켠 로지스틱 회귀의 혼동행렬")
    pred_lr = (probas["로지스틱 회귀"] >= 0.5).astype(int)
    tn, fp, fn, tp = cm_counts(y, pred_lr)
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(cm_figure(tn, fp, fn, tp, "로지스틱 회귀 (판정 기준 0.5)"),
                        use_container_width=True)
    with right:
        st.markdown(f"""
- **TP {tp}**: 실제 많은 해를 '많음'으로 맞힘
- **FP {fp}**: 적은 해인데 '많음'이라 잘못 예측 (헛경보)
- **FN {fn}**: 많은 해인데 '적음'이라 놓침
- **TN {tn}**: 적은 해를 '적음'으로 맞힘
""")

    st.header("2️⃣ 정밀도·재현율·F1이란? (숫자로 직접 계산)")
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    st.markdown(f"""
- **정밀도** = '많음'이라고 한 것 중 진짜 많은 해의 비율 = TP ÷ (TP+FP)
  = {tp} ÷ ({tp}+{fp}) = **{precision:.3f}**
- **재현율** = 진짜 많은 해 중 찾아낸 비율 = TP ÷ (TP+FN)
  = {tp} ÷ ({tp}+{fn}) = **{recall:.3f}**
- **F1** = 정밀도와 재현율의 조화평균 = 2×정밀도×재현율 ÷ (정밀도+재현율)
  = 2×{precision:.3f}×{recall:.3f} ÷ ({precision:.3f}+{recall:.3f}) = **{f1:.3f}**
- **정확도** = (TP+TN) ÷ 전체 = ({tp}+{tn}) ÷ {tp + tn + fp + fn} = **{acc:.3f}**
""")
    st.caption("분모가 0이면 0으로 처리했어요. 이 값들이 아래 표의 로지스틱 회귀 값과 같은지 확인해 보세요.")

    st.header("3️⃣ 정확도 비교: 가중치 켬 vs 끔 vs 베이스라인")
    pred_off = (proba_off >= 0.5).astype(int)
    acc_df = pd.DataFrame({
        "모델": ["로지스틱(가중치 켬)", "로지스틱(가중치 끔)", "베이스라인(항상 다수 클래스)"],
        "정확도": [accuracy_score(y, pred_lr), accuracy_score(y, pred_off),
                accuracy_score(y, base_pred)],
        "재현율": [recall_score(y, pred_lr, zero_division=0),
                recall_score(y, pred_off, zero_division=0),
                recall_score(y, base_pred, zero_division=0)],
    })
    st.dataframe(acc_df.style.format({"정확도": "{:.3f}", "재현율": "{:.3f}"}),
                 hide_index=True, use_container_width=True)
    st.caption("베이스라인은 '많음'이 소수일 때 모두 '적음'이라고만 해도 정확도가 높게 나옵니다. "
               "그래서 정확도만 보면 안 되고 재현율을 함께 봐야 합니다.")

    st.header("4️⃣ 세 모델 + 베이스라인 비교 (가중치 켬)")
    rows = [{"모델": "베이스라인", **cls_metrics(y, base_pred)}]
    for n, p in probas.items():
        rows.append({"모델": n, **cls_metrics(y, (p >= 0.5).astype(int))})
    comp = pd.DataFrame(rows)
    metric_cols = ["정확도", "정밀도", "재현율", "F1"]
    st.dataframe(comp.style.format({c: "{:.3f}" for c in metric_cols}),
                 hide_index=True, use_container_width=True)
    long = comp.melt(id_vars="모델", var_name="지표", value_name="점수")
    fig = px.bar(long, x="지표", y="점수", color="모델", barmode="group",
                 text=long["점수"].round(2))
    fig.update_layout(yaxis_range=[0, 1.1])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("최종 모델 고르기")
    crit = st.radio("선택 기준", ["F1 우선 (정밀도·재현율 균형)", "재현율 우선 (많은 해를 놓치지 않기)"],
                    horizontal=True)
    key = "F1" if crit.startswith("F1") else "재현율"
    sec = "재현율" if key == "F1" else "F1"
    real = comp[comp["모델"] != "베이스라인"].sort_values([key, sec], ascending=False)
    top = real.iloc[0]
    st.success(f"기준 '{key}'으로 고르면 **{top['모델']}** (F1 {top['F1']:.3f}, "
               f"재현율 {top['재현율']:.3f}, 정밀도 {top['정밀도']:.3f}, 정확도 {top['정확도']:.3f})")
    st.markdown("""
**최종 모델을 고를 때 기준 예시**
1. 베이스라인보다 재현율·F1이 나은가? (정확도는 참고만)
2. 사고가 많은 해를 놓치는 것(FN)과 헛경보(FP) 중 무엇이 더 큰 문제인가? 놓치면 안 된다면 재현율 우선
3. 점수가 비슷하면 더 단순하고 설명하기 쉬운 모델 (로지스틱 회귀 > 결정트리 > 랜덤 포레스트)
4. 표본이 16개라 점수 차이가 1~2개 연도 때문일 수 있음을 함께 적기
""")

    st.header("5️⃣ 판정 기준을 바꾸면?")
    model_name = st.selectbox("모델 선택", list(models.keys()))
    thr = st.slider("판정 기준 (확률이 이 값 이상이면 '많음')", 0.05, 0.95, 0.5, 0.05)
    p = probas[model_name]
    pred_t = (p >= thr).astype(int)
    tn2, fp2, fn2, tp2 = cm_counts(y, pred_t)
    m = cls_metrics(y, pred_t)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("정확도", f"{m['정확도']:.3f}")
    k2.metric("정밀도", f"{m['정밀도']:.3f}")
    k3.metric("재현율", f"{m['재현율']:.3f}")
    k4.metric("F1", f"{m['F1']:.3f}")

    a, b = st.columns(2)
    with a:
        st.plotly_chart(cm_figure(tn2, fp2, fn2, tp2, f"{model_name} (기준 {thr})"),
                        use_container_width=True)
    with b:
        ths = np.round(np.arange(0.05, 0.96, 0.05), 2)
        rows = []
        for t in ths:
            mm = cls_metrics(y, (p >= t).astype(int))
            rows.append({"기준": t, "정밀도": mm["정밀도"], "재현율": mm["재현율"], "F1": mm["F1"]})
        tdf = pd.DataFrame(rows).melt(id_vars="기준", var_name="지표", value_name="점수")
        fig = px.line(tdf, x="기준", y="점수", color="지표", markers=True,
                      title="판정 기준에 따른 정밀도·재현율·F1")
        fig.add_vline(x=thr, line_dash="dash", line_color="gray")
        fig.update_layout(yaxis_range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)
    st.caption("기준을 낮추면 '많음'이라고 더 쉽게 말해 재현율은 오르고 정밀도는 내려가는 경향이 있어요. "
               "트리 모델은 확률이 몇 가지 값뿐이라 선이 계단처럼 변할 수 있어요.")

    st.warning("자료가 16개 연도뿐이라 한두 해만 달라져도 지표가 크게 변합니다. "
               "지정 수와 사고의 관계는 상관일 뿐 원인이라고 단정할 수 없습니다.")


# =============================================================
# 메인
# =============================================================
data, log_df, err = load_data()
if err:
    st.error(err)
    st.stop()

page = st.sidebar.radio("📂 페이지 선택",
                        ["🚸 교통안전 분석", "📐 회귀 모델 비교", "🎯 분류 모델 비교"])
st.sidebar.divider()

if page.startswith("🚸"):
    page_analysis(data, log_df)
elif page.startswith("📐"):
    page_regression(data)
else:
    page_classification(data)
