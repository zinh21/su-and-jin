import glob
import re
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.tree import DecisionTreeRegressor

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
    train, test = df[df["연도"] < split_year], df[df["연도"] >= split_year]
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    st.caption(f"학습 {len(train)}개({train['연도'].min()}~{train['연도'].max()}) / "
               f"테스트 {len(test)}개({test['연도'].min()}~{test['연도'].max()}) - 시간 순서 분할")

    models = {
        "베이스라인(평균)": DummyRegressor(strategy="mean"),
        "선형 회귀": LinearRegression(),
        "결정트리": DecisionTreeRegressor(max_depth=max_depth, random_state=42),
        "랜덤 포레스트": RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth,
                                            random_state=42),
    }

    rows, preds = [], {}
    X_all, y_all = df[features], df[target]
    for name, model in models.items():
        model.fit(X_train, y_train)
        p = model.predict(X_test)
        preds[name] = p
        loo = cross_val_predict(model, X_all, y_all, cv=LeaveOneOut())
        rows.append({
            "모델": name,
            "MAE": mean_absolute_error(y_test, p),
            "RMSE": np.sqrt(mean_squared_error(y_test, p)),
            "R²": r2_score(y_test, p) if len(y_test) > 1 else np.nan,
            "LOOCV MAE": mean_absolute_error(y_all, loo),
            "LOOCV R²": r2_score(y_all, loo),
        })

    result = pd.DataFrame(rows)
    cols = [c for c in result.columns if c != "모델"]

    st.header("📋 성능 비교표")
    st.dataframe(result.style.format({c: "{:.3f}" for c in cols}),
                 hide_index=True, use_container_width=True)
    st.caption("MAE·RMSE는 작을수록, R²는 1에 가까울수록 좋습니다. R²가 0보다 작으면 평균 예측보다 못합니다.")

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
    best = real.loc[real["MAE"].idxmin()]
    base = result[result["모델"] == "베이스라인(평균)"].iloc[0]
    st.info(f"🔎 **발견:** 테스트 MAE가 가장 낮은 모델은 **{best['모델']}**입니다.  \n"
            f"📎 **근거:** MAE {best['MAE']:.2f} (베이스라인 {base['MAE']:.2f}), "
            f"R² {best['R²']:.3f}, LOOCV R² {best['LOOCV R²']:.3f}")

    st.header("🔍 선형 회귀 해석")
    lin = models["선형 회귀"]
    for f, c in zip(features, lin.coef_):
        st.write(f"`{f}` 1 증가 시 {target} 변화: **{c:+.4f}**")
    st.write(f"절편: {lin.intercept_:.2f}")

    st.header("📝 README에 붙여넣을 표")
    head = "| " + " | ".join(result.columns) + " |"
    div = "|" + "|".join(["---"] * len(result.columns)) + "|"
    body = ["| " + " | ".join([r["모델"]] + [f"{r[c]:.3f}" for c in cols]) + " |"
            for _, r in result.iterrows()]
    md = "\n".join([head, div] + body)
    st.code(md, language="markdown")
    st.download_button("표를 .md로 저장", md, file_name="regression_comparison.md")

    st.warning("표본이 16개뿐이고 지정 수와 사고 건수는 둘 다 시간에 따라 변해 상관이 있어도 "
               "원인이라 할 수 없습니다. 트리 계열은 학습 범위 밖 값을 예측하지 못합니다.")


# =============================================================
# 메인
# =============================================================
data, log_df, err = load_data()
if err:
    st.error(err)
    st.stop()

page = st.sidebar.radio("📂 페이지 선택", ["🚸 교통안전 분석", "📐 회귀 모델 비교"])
st.sidebar.divider()

if page.startswith("🚸"):
    page_analysis(data, log_df)
else:
    page_regression(data)
