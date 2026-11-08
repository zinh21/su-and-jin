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
# 페이지 3: 검증 기록 (log)
# =============================================================
def page_log(df):
    st.title("📓 검증 기록 남기기")
    st.markdown("AI의 제안을 **시험용 데이터로 확인**해 고치거나 버린 과정을 log에 남깁니다. "
                "아래 표는 앱이 계산한 실제 값이니 근거로 쓰세요.")

    min_y, max_y = int(df["연도"].min()), int(df["연도"].max())
    c1, c2, c3 = st.columns(3)
    target = c1.selectbox("예측할 값(y)", [c for c in ["사고 건수", "사망자", "부상자"]
                                          if df[c].notna().any()])
    split_year = c2.slider("테스트 시작 연도", min_y + 8, max_y - 1, min(2021, max_y - 1))
    max_depth = c3.slider("트리 max_depth", 1, 6, 2)

    result, _, _, train, test = evaluate(df, target, ["지정 수"], split_year, max_depth, 100)
    st.caption(f"학습 {train['연도'].min()}~{train['연도'].max()} ({len(train)}개) / "
               f"시험 {test['연도'].min()}~{test['연도'].max()} ({len(test)}개)")

    st.subheader("근거: 학습용 vs 시험용 지표")
    cols = [c for c in result.columns if c != "모델"]
    st.dataframe(result.style.format({c: "{:.3f}" for c in cols}),
                 hide_index=True, use_container_width=True)
    st.caption("분류의 '정확도 1.0'처럼, 회귀에서는 학습 R²가 1에 가깝고 시험 R²가 낮으면 의심하세요.")

    st.subheader("기록 입력")
    name = st.text_input("학번 이름", "")
    decision = st.text_area(
        "오늘 결정한 것",
        "모델은 학습용이 아니라 시험용 데이터의 MAE·RMSE·R²로 비교하기로 함.")
    prompt = st.text_area(
        "AI에게 보낸 프롬프트(원문)",
        "학습 R²가 1에 가까운 게 이상해. 결정트리는 깊게 만들면 학습 데이터를 외운다고 했잖아. "
        "이 값은 어떤 데이터로 계산한 거야? 시험용 데이터로 MAE, RMSE, R²를 다시 계산해 줘.")
    discard = st.text_area(
        "AI가 준 것 중 고치거나 버린 것과 그 이유",
        "AI는 ○○ 모델이 가장 좋다고 했지만 학습용으로 계산한 값이었고, "
        "시험용으로 다시 계산하니 시험 R²가 ○○로 낮아 그 결론을 버림. (위 표의 실제 숫자로 바꾸세요)")
    feeling = st.text_area(
        "느낀 점",
        "학습용 점수가 너무 좋다는 것이 오히려 의심해야 할 신호라는 걸 알게 됨.")

    log_md = f"""## {name if name else '학번 이름'}
- 오늘 결정한 것: {decision}
- AI에게 보낸 프롬프트(원문): {prompt}
- AI가 준 것 중 고치거나 버린 것과 그 이유: {discard}
- 느낀 점: {feeling}

**근거 표** (테스트 시작 연도 {split_year}, 예측값 {target})

{to_markdown_table(result)}
"""
    st.subheader("log에 붙여넣을 기록")
    st.code(log_md, language="markdown")
    st.download_button("기록을 .md로 저장", log_md, file_name="verification_log.md")
    st.warning("프롬프트는 실제로 보낸 원문을 쓰고, 숫자는 위 표에서 가져오세요. "
               "다른 사람의 결과(예: 재현율 0~0.16)를 그대로 옮기면 안 됩니다.")


# =============================================================
# 메인
# =============================================================
data, log_df, err = load_data()
if err:
    st.error(err)
    st.stop()

page = st.sidebar.radio("📂 페이지 선택",
                        ["🚸 교통안전 분석", "📐 회귀 모델 비교", "📓 검증 기록"])
st.sidebar.divider()

if page.startswith("🚸"):
    page_analysis(data, log_df)
elif page.startswith("📐"):
    page_regression(data)
else:
    page_log(data)
