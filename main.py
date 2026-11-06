import csv
import re
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------
# 페이지 설정 (한 번만)
# -----------------------------
st.set_page_config(
    page_title="당곡고 데이터 분석 도우미",
    page_icon="📊",
    layout="wide",
)

ZONE_URL = "https://raw.githubusercontent.com/zinh21/su-and-jin/refs/heads/main/161601_20261026083839280_excel.csv"
TRAFFIC_URL = "https://raw.githubusercontent.com/zinh21/su-and-jin/refs/heads/main/%EA%B5%90%ED%86%B5%EC%82%AC%EA%B3%A0%ED%86%B5%EA%B3%84_20261026.csv"

TOTAL_WORDS = ["합계", "전국", "계", "전체", "총계"]
YEAR_RE = r"^\s*((?:19|20)\d{2})"
MISSING_MARKS = {"", "-", "–", "—", "…", "N/A", "n/a", "NA", "nan", "NaN", "None", "해당없음", "없음"}
FFILL_NAME_HINTS = ["시도", "연도", "구분", "항목", "분류", "지역"]

clean_log = []
removed_missing = {}


# =============================================================
# 공통/어린이보호구역 분석용 기본 함수
# =============================================================
def to_number(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def to_year(series):
    return pd.to_numeric(series.astype(str).str.extract(r"((?:19|20)\d{2})")[0], errors="coerce")


def is_text(series):
    return not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def pct(a, b):
    return (a - b) / b * 100 if b else 0


def make_unique(columns):
    seen, result = {}, []
    for c in columns:
        c = str(c).strip()
        if c in seen:
            seen[c] += 1
            result.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            result.append(c)
    return result


def clean_item(text):
    t = re.sub(YEAR_RE, "", str(text))
    t = re.sub(r"[._]\d+$", "", t)
    t = t[1:] if t.startswith("년") else t
    return t.strip()


def overview(df):
    return pd.DataFrame({
        "열 이름": [str(c) for c in df.columns],
        "자료형": [str(t) for t in df.dtypes],
        "빈 값 개수": df.isna().sum().values,
        "빈 값 비율(%)": (df.isna().mean() * 100).round(1).values,
    })


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_bytes(url):
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        return res.content
    except Exception:
        return None


def read_plain(content):
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(BytesIO(content), encoding=enc)
        except Exception:
            continue
    return None


def fix_structure(content):
    text = None
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            text = content.decode(enc)
            break
        except Exception:
            continue
    if text is None:
        return None, {}

    rows = list(csv.reader(StringIO(text)))
    if not rows:
        return None, {}

    width = max(len(r) for r in rows)
    rows = [[c.strip() for c in r] + [""] * (width - len(r)) for r in rows]
    raw = pd.DataFrame(rows).replace(list(MISSING_MARKS), pd.NA)

    info = {"원본 줄 수": len(raw)}

    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)

    max_filled = raw.head(10).notna().sum(axis=1).max()
    header_idx = 0
    for i in range(min(10, len(raw))):
        if raw.iloc[i].notna().sum() >= max(2, 0.6 * max_filled):
            header_idx = i
            break
    info["건너뛴 제목 줄 수"] = header_idx

    header = raw.iloc[header_idx].fillna("").astype(str).tolist()
    header = [h if h else f"열{i + 1}" for i, h in enumerate(header)]
    df = raw.iloc[header_idx + 1:].reset_index(drop=True)
    df.columns = make_unique(header)

    filled = []
    text_cols = [c for c in df.columns
                 if df[c].notna().any() and to_number(df[c].dropna()).notna().mean() < 0.5]
    for i, c in enumerate(text_cols):
        if i == 0 or any(h in c for h in FFILL_NAME_HINTS):
            before = int(df[c].isna().sum())
            df[c] = df[c].ffill()
            if before - int(df[c].isna().sum()) > 0:
                filled.append(c)
    info["병합 셀로 보고 채운 열"] = ", ".join(filled) if filled else "없음"

    for c in df.columns:
        nn = df[c].dropna()
        if len(nn) and to_number(nn).notna().mean() >= 0.9:
            df[c] = to_number(df[c])

    info["정리 후 행 수"] = len(df)
    return df, info


def to_long(df, name):
    df = df.copy()
    df.columns = make_unique(df.columns)
    year_cols = [c for c in df.columns if re.match(YEAR_RE, c)]

    if len(year_cols) >= 1:
        items = {c: clean_item(c) for c in year_cols}
        if len(df):
            first = df.iloc[0][year_cols]
            is_txt = to_number(first).isna() & first.notna() & first.astype(str).str.strip().ne("")
            if is_txt.mean() > 0.5:
                items = {c: str(first[c]).strip() for c in year_cols}
                df = df.iloc[1:].reset_index(drop=True)
        id_cols = [c for c in df.columns if c not in year_cols]
        parts = []
        for c in year_cols:
            tmp = df[id_cols].copy()
            tmp["__연도"] = int(re.match(YEAR_RE, c).group(1))
            tmp["__항목"] = items[c] if items[c] else "값"
            tmp["__값"] = to_number(df[c])
            parts.append(tmp)
        long = pd.concat(parts, ignore_index=True)
    else:
        ycol = None
        for c in df.columns:
            if to_year(df[c]).notna().mean() > 0.6:
                ycol = c
                break
        if ycol is None:
            return None
        value_cols = [c for c in df.columns if c != ycol and to_number(df[c]).notna().mean() > 0.5]
        id_cols = [c for c in df.columns if c != ycol and c not in value_cols]
        parts = []
        for c in value_cols:
            tmp = df[id_cols].copy()
            tmp["__연도"] = to_year(df[ycol])
            tmp["__항목"] = c.strip()
            tmp["__값"] = to_number(df[c])
            parts.append(tmp)
        if not parts:
            return None
        long = pd.concat(parts, ignore_index=True)

    before = len(long)
    missing_mask = long["__연도"].isna() | long["__값"].isna()
    removed_missing[name] = long[missing_mask].copy()
    long = long[~missing_mask].copy()
    long["__연도"] = long["__연도"].astype(int)
    clean_log.append({
        "자료": name, "단계": "결측치 처리",
        "처리 방법": "숫자가 아니거나 빈 값인 행 삭제",
        "처리 전 행 수": before, "삭제된 행 수": before - len(long), "처리 후 행 수": len(long),
    })
    return long


def keep_total_rows(df):
    for c in df.columns:
        if c.startswith("__") or not is_text(df[c]):
            continue
        mask = df[c].astype(str).str.strip().isin(TOTAL_WORDS)
        if mask.any():
            df = df[mask]
    return df


def yearly(df):
    if df is None or df.empty:
        return None
    return df.groupby("__연도")["__값"].sum().sort_index()


def pick(long, keywords, exclude_kw=()):
    if long is None:
        return None
    norm = lambda s: str(s).replace(" ", "")
    for kw in keywords:
        for item in long["__항목"].unique():
            n = norm(item)
            if kw in n and not any(e in n for e in exclude_kw):
                s = yearly(keep_total_rows(long[long["__항목"] == item]))
                if s is not None:
                    return s
    for kw in keywords:
        for c in long.columns:
            if c.startswith("__") or not is_text(long[c]):
                continue
            vals = long[c].astype(str).map(norm)
            mask = vals.str.contains(kw, regex=False)
            for e in exclude_kw:
                mask &= ~vals.str.contains(e, regex=False)
            if mask.any():
                s = yearly(keep_total_rows(long[mask]))
                if s is not None:
                    return s
    return None


def show_structure(title, df):
    st.write(f"**{title}** 열 이름:", list(df.columns))
    st.dataframe(df.head(10), use_container_width=True)
    for c in df.columns:
        if is_text(df[c]):
            st.write(f"`{c}` 열의 값 예시:", list(df[c].dropna().astype(str).unique()[:15]))


def finding(df, col, label, unit):
    s = df[col].reset_index(drop=True)
    y = df["연도"].reset_index(drop=True)
    first_v, last_v = s.iloc[0], s.iloc[-1]
    d = last_v - first_v
    word = "증가" if d > 0 else "감소" if d < 0 else "변화 없음"
    peak_y, low_y = int(y[s.idxmax()]), int(y[s.idxmin()])
    return (
        f"🔎 **발견:** {label}은(는) {int(y.iloc[0])}년 대비 {int(y.iloc[-1])}년에 **{word}**했고 "
        f"최댓값은 {peak_y}년입니다.  \n"
        f"📎 **근거:** {int(y.iloc[0])}년 {first_v:,.0f}{unit} → {int(y.iloc[-1])}년 {last_v:,.0f}{unit} "
        f"({d:+,.0f}{unit}, {pct(last_v, first_v):+.1f}%), "
        f"최대 {s.max():,.0f}{unit}({peak_y}년), 최소 {s.min():,.0f}{unit}({low_y}년)"
    )


# =============================================================
# 페이지 1: 어린이보호구역 교통안전 분석
# =============================================================
def run_traffic_app():
    clean_log.clear()
    removed_missing.clear()

    st.title("🚸 어린이보호구역 교통안전 분석")
    st.markdown("어린이보호구역 지정 현황과 교통사고 변화를 한눈에 살펴봅니다.")

    zone_bytes = load_bytes(ZONE_URL)
    traffic_bytes = load_bytes(TRAFFIC_URL)

    if zone_bytes is None or traffic_bytes is None:
        st.error("CSV 파일을 불러오지 못했습니다. 인터넷 연결과 주소(공개 저장소인지)를 확인해 주세요.")
        return

    zone_plain, traffic_plain = read_plain(zone_bytes), read_plain(traffic_bytes)
    zone_raw, zone_info = fix_structure(zone_bytes)
    traffic_raw, traffic_info = fix_structure(traffic_bytes)

    if zone_raw is None or traffic_raw is None:
        st.error("CSV 구조를 정리하지 못했습니다. 파일을 확인해 주세요.")
        return

    # ---- 데이터 개요 ----
    st.header("🗂️ 데이터 개요")
    st.caption("파일 구조를 정리한 뒤의 값입니다. 정리 전 빈 값과 비교해 보세요.")

    for title, plain, fixed, info in [
        ("어린이보호구역 CSV", zone_plain, zone_raw, zone_info),
        ("교통사고통계 CSV", traffic_plain, traffic_raw, traffic_info),
    ]:
        st.subheader(title)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("행 개수", f"{fixed.shape[0]:,}")
        m2.metric("열 개수", f"{fixed.shape[1]:,}")
        before_na = int(plain.isna().sum().sum()) if plain is not None else 0
        after_na = int(fixed.isna().sum().sum())
        m3.metric("빈 값 (정리 전)", f"{before_na:,}")
        m4.metric("빈 값 (정리 후)", f"{after_na:,}", delta=f"{after_na - before_na:+,}", delta_color="inverse")
        m5.metric("건너뛴 제목 줄", info.get("건너뛴 제목 줄 수", 0))
        st.write(f"병합 셀로 보고 채운 열: **{info.get('병합 셀로 보고 채운 열', '없음')}**")
        st.dataframe(overview(fixed), hide_index=True, use_container_width=True)

    st.divider()

    # ---- 결측치 처리 ----
    zone_long = to_long(zone_raw, "어린이보호구역")
    traffic_long = to_long(traffic_raw, "교통사고")

    st.header("🧹 결측치 처리 결과")
    if clean_log:
        st.dataframe(pd.DataFrame(clean_log), hide_index=True, use_container_width=True)
    st.caption("결측치는 숫자가 아니거나 비어 있는 값의 행을 삭제했습니다. 이상치는 처리하지 않았습니다.")

    for name in ["어린이보호구역", "교통사고"]:
        with st.expander(f"🗑️ {name}: 삭제된 행 보기"):
            miss_df = removed_missing.get(name)
            st.write(f"결측치로 삭제된 행: {0 if miss_df is None else len(miss_df):,}개 (앞 50개만 표시)")
            if miss_df is not None and len(miss_df):
                st.dataframe(miss_df.head(50), use_container_width=True)

    st.divider()

    # ---- 어린이보호구역 지정 수 ----
    zone_series = pick(zone_long, ["지정", "개소", "보호구역"])
    if zone_series is None and zone_long is not None and not zone_long.empty:
        first_item = zone_long["__항목"].iloc[0]
        zone_series = yearly(keep_total_rows(zone_long[zone_long["__항목"] == first_item]))

    if zone_series is None or len(zone_series) == 0:
        st.error("어린이보호구역 CSV에서 연도/지정 수를 찾지 못했습니다. 아래 구조를 알려주세요.")
        show_structure("어린이보호구역 CSV (정리 후)", zone_raw)
        return

    zones = zone_series.rename("지정 수").reset_index().rename(columns={"__연도": "연도"})

    # ---- 교통사고 ----
    death = pick(traffic_long, ["사망"])
    injury = pick(traffic_long, ["부상"])
    acc = pick(traffic_long, ["사고건수", "발생건수", "사고", "건수"], exclude_kw=["사망", "부상"])

    found = {}
    for name, s in [("사고 건수", acc), ("사망자 수", death), ("부상자 수", injury)]:
        if s is not None and len(s) > 0:
            found[name] = s

    if not found:
        st.error("교통사고 CSV에서 사고/사망/부상 항목을 찾지 못했습니다. 아래 구조를 알려주세요.")
        show_structure("교통사고 CSV (정리 후)", traffic_raw)
        return

    traffic = pd.DataFrame(found).sort_index().reset_index()
    traffic = traffic.rename(columns={traffic.columns[0]: "연도"})
    items = list(found.keys())

    common = sorted(set(zones["연도"]) & set(traffic["연도"]))
    if len(common) >= 1:
        zones = zones[zones["연도"].isin(common)].reset_index(drop=True)
        traffic = traffic[traffic["연도"].isin(common)].reset_index(drop=True)

    first_y, last_y = int(traffic["연도"].iloc[0]), int(traffic["연도"].iloc[-1])
    st.caption(f"분석 기간: {first_y}년 ~ {last_y}년")

    with st.expander("🔧 구조 정리 후 CSV 확인 (문제가 있을 때 열어 보세요)"):
        show_structure("어린이보호구역 CSV (정리 후)", zone_raw)
        show_structure("교통사고 CSV (정리 후)", traffic_raw)

    # ---- 핵심 수치 ----
    st.header("📌 핵심 데이터")
    cols = st.columns(1 + len(items))
    z_last, z_first = zones["지정 수"].iloc[-1], zones["지정 수"].iloc[0]
    cols[0].metric(
        f"{int(zones['연도'].iloc[-1])}년 보호구역 지정 수",
        f"{z_last:,.0f}개",
        delta=f"{z_last - z_first:+,.0f}개 ({int(zones['연도'].iloc[0])}년 대비)",
    )
    for i, name in enumerate(items):
        last = traffic[name].iloc[-1]
        delta = f"{last - traffic[name].iloc[-2]:+,.0f} (전년 대비)" if len(traffic) > 1 else None
        cols[i + 1].metric(f"{last_y}년 {name}", f"{last:,.0f}", delta=delta, delta_color="inverse")

    st.divider()

    # ---- 그래프 + 발견 문장 ----
    st.header("📊 그래프와 발견한 점")

    def bar(df, col, title, color):
        fig = px.bar(df, x="연도", y=col, text=col, title=title, color_discrete_sequence=[color])
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig.update_layout(xaxis=dict(type="category"), yaxis_title=None)
        return fig

    colors = {"사고 건수": "#EF553B", "사망자 수": "#636EFA", "부상자 수": "#FFA15A"}
    units = {"지정 수": "개", "사고 건수": "건", "사망자 수": "명", "부상자 수": "명"}
    charts = [("지정 수", zones, "연도별 어린이보호구역 지정 수", "#00CC96", "보호구역 지정 수")]
    charts += [(n, traffic, f"연도별 {n}", colors[n], n) for n in items]

    for i in range(0, len(charts), 2):
        row = st.columns(2)
        for j, (col, df, title, color, label) in enumerate(charts[i:i + 2]):
            with row[j]:
                st.plotly_chart(bar(df, col, title, color), use_container_width=True)
                st.info(finding(df, col, label, units[col]))

    st.subheader("📈 시작 연도를 100으로 놓고 본 변화율 비교")
    merged = pd.merge(zones, traffic, on="연도", how="inner")
    if len(merged) >= 2:
        idx = pd.DataFrame({"연도": merged["연도"]})
        for n in ["지정 수"] + items:
            base = merged[n].iloc[0]
            label = "보호구역 지정 수" if n == "지정 수" else n
            idx[label] = merged[n] / base * 100 if base else None
        idx_long = idx.melt(id_vars="연도", var_name="항목", value_name="지수")
        fig_idx = px.line(idx_long, x="연도", y="지수", color="항목", markers=True,
                          title=f"{int(merged['연도'].iloc[0])}년 = 100 기준 변화")
        fig_idx.update_layout(xaxis=dict(type="category"))
        st.plotly_chart(fig_idx, use_container_width=True)

        last_idx = idx.iloc[-1].drop("연도").astype(float)
        detail = ", ".join(f"{k} {v:.1f}" for k, v in last_idx.items())
        st.info(
            f"🔎 **발견:** {int(idx['연도'].iloc[-1])}년 기준 가장 많이 오른(또는 덜 줄어든) 항목은 "
            f"**{last_idx.idxmax()}**, 가장 많이 줄어든 항목은 **{last_idx.idxmin()}**입니다.  \n"
            f"📎 **근거:** 시작 연도=100일 때 {detail}"
        )
    else:
        st.info("두 자료에 겹치는 연도가 2개 미만이라 변화율 비교는 생략했습니다.")

    st.divider()

    st.header("📋 정리된 데이터 표")
    st.dataframe(merged if not merged.empty else traffic, hide_index=True, use_container_width=True)

    st.warning(
        "주의: 이 데이터만으로 보호구역 지정 수의 변화가 교통사고 변화의 원인이라고 단정할 수 없습니다. "
        "차량 통행량, 보호구역별 사고율, 안전시설 변화 등의 자료를 함께 분석해야 합니다."
    )

    st.divider()
    st.header("💡 더 탐구해 볼 질문")
    st.markdown("""
- 삭제된 결측치 행은 어떤 지역·연도·항목에 몰려 있을까?
- 남은 빈 값은 '0건'일까, '자료 없음'일까?
- 이상치를 따로 처리하지 않았을 때 결과 해석에 어떤 영향이 있을까?
- 사고 건수뿐 아니라 차량 통행량을 고려하면 결론이 달라질까?
""")

    st.caption("자료: 제공된 어린이보호구역 및 교통사고통계 CSV 파일")


# =============================================================
# 페이지 2: Stroke 예측 (로지스틱 회귀)
# =============================================================
def run_stroke_app():
    st.title("📈 Stroke 예측: 로지스틱 회귀 모델")
    st.markdown("로지스틱 회귀를 학습하고, 베이스라인과 비교하며 결과를 해석합니다.")

    # ---- 1. 데이터 불러오기 ----
    uploaded = st.file_uploader(
        "stroke 데이터 CSV 업로드 (예: healthcare-dataset-stroke-data.csv)", type="csv"
    )
    if uploaded is None:
        st.info("CSV 파일을 업로드하면 분석이 시작됩니다.")
        return

    df = pd.read_csv(uploaded)
    st.dataframe(df.head(), use_container_width=True)

    target = st.selectbox(
        "예측할 열(타깃)",
        df.columns,
        index=list(df.columns).index("stroke") if "stroke" in df.columns else len(df.columns) - 1,
    )

    df = df.drop(columns=[c for c in df.columns if c.lower() == "id"])
    df = df.dropna(subset=[target])
    y = df[target]
    X = df.drop(columns=[target])

    if y.nunique() != 2:
        st.error("타깃 열은 값이 2종류(있음/없음)여야 합니다.")
        return

    classes = sorted(y.unique())
    positive = classes[-1]

    # ---- 2. 설정 ----
    st.sidebar.header("⚙️ 로지스틱 회귀 설정")
    test_size = st.sidebar.slider("테스트 데이터 비율", 0.1, 0.4, 0.2, 0.05)
    seed = int(st.sidebar.number_input("random_state", value=42, step=1))
    use_weight = st.sidebar.checkbox("class_weight='balanced' 적용", value=True)
    C = st.sidebar.select_slider("규제 강도 C (작을수록 강한 규제)",
                                 options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
    threshold = st.sidebar.slider("분류 임계값 (확률이 이 값 이상이면 '있음')", 0.05, 0.95, 0.5, 0.05)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    st.caption(f"학습 {len(X_train):,}개 / 테스트 {len(X_test):,}개 (층화 추출)")

    # ---- 3. 전처리 + 모델 ----
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

    logit = Pipeline([
        ("prep", prep),
        ("model", LogisticRegression(C=C, max_iter=1000,
                                     class_weight="balanced" if use_weight else None)),
    ])
    baseline = Pipeline([
        ("prep", prep),
        ("model", DummyClassifier(strategy="most_frequent")),
    ])

    logit.fit(X_train, y_train)
    baseline.fit(X_train, y_train)

    proba = logit.predict_proba(X_test)[:, list(logit.classes_).index(positive)]
    pred = np.where(proba >= threshold, positive, classes[0])
    base_pred = baseline.predict(X_test)

    # ---- 4. 성능 비교 ----
    def calc_metrics(true, p, score=None):
        return {
            "정확도": accuracy_score(true, p),
            "재현율": recall_score(true, p, pos_label=positive, zero_division=0),
            "정밀도": precision_score(true, p, pos_label=positive, zero_division=0),
            "F1": f1_score(true, p, pos_label=positive, zero_division=0),
            "ROC-AUC": roc_auc_score(true == positive, score) if score is not None else 0.5,
        }

    result = pd.DataFrame([
        {"모델": "베이스라인(항상 다수 클래스)", **calc_metrics(y_test, base_pred)},
        {"모델": f"로지스틱 회귀 (임계값 {threshold})", **calc_metrics(y_test, pred, proba)},
    ])

    st.header("📊 성능 비교")
    st.dataframe(result.style.format({c: "{:.3f}" for c in result.columns if c != "모델"}),
                 hide_index=True, use_container_width=True)

    long = result.melt(id_vars="모델", var_name="지표", value_name="점수")
    fig = px.bar(long, x="지표", y="점수", color="모델", barmode="group",
                 text=long["점수"].round(3), title="베이스라인 vs 로지스틱 회귀")
    fig.update_layout(yaxis_range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

    m, b = result.iloc[1], result.iloc[0]
    st.info(
        f"🔎 **발견:** 로지스틱 회귀의 재현율은 **{m['재현율']:.3f}**, 베이스라인은 **{b['재현율']:.3f}**입니다.  \n"
        f"📎 **근거:** 정확도 {b['정확도']:.3f} → {m['정확도']:.3f}, "
        f"정밀도 {m['정밀도']:.3f}, ROC-AUC {m['ROC-AUC']:.3f}"
    )

    # ---- 5. 혼동행렬 ----
    st.header("🧮 혼동행렬 (행: 실제, 열: 예측)")
    cm = confusion_matrix(y_test, pred, labels=classes)
    st.dataframe(
        pd.DataFrame(cm, index=[f"실제 {c}" for c in classes],
                     columns=[f"예측 {c}" for c in classes]),
        use_container_width=True,
    )

    # ---- 6. ROC 곡선 ----
    st.header("📉 ROC 곡선")
    fpr, tpr, _ = roc_curve(y_test == positive, proba)
    roc_df = pd.DataFrame({"거짓 양성률(FPR)": fpr, "재현율(TPR)": tpr})
    fig_roc = px.line(roc_df, x="거짓 양성률(FPR)", y="재현율(TPR)", title="ROC 곡선")
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    st.plotly_chart(fig_roc, use_container_width=True)
    st.caption("점선(대각선)은 무작위 추측입니다. 곡선이 왼쪽 위로 휠수록 좋은 모델입니다.")

    # ---- 7. 계수 해석 ----
    st.header("🔍 계수로 보는 영향 요인")
    feature_names = logit.named_steps["prep"].get_feature_names_out()
    coefs = logit.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame({"특성": feature_names, "계수": coefs})
    coef_df["오즈비(e^계수)"] = np.exp(coef_df["계수"])
    coef_df["영향 방향"] = np.where(coef_df["계수"] > 0, "위험 ↑", "위험 ↓")
    coef_df = coef_df.reindex(coef_df["계수"].abs().sort_values(ascending=False).index)

    fig_coef = px.bar(coef_df.head(15).sort_values("계수"), x="계수", y="특성", orientation="h",
                      color="영향 방향", title="영향이 큰 특성 상위 15개")
    st.plotly_chart(fig_coef, use_container_width=True)
    st.dataframe(coef_df.head(15).style.format({"계수": "{:.3f}", "오즈비(e^계수)": "{:.3f}"}),
                 hide_index=True, use_container_width=True)
    st.caption("수치형 특성은 표준화되어 있어 계수 크기를 서로 비교할 수 있습니다. "
               "계수가 크다고 원인이라는 뜻은 아니며, 상관관계일 뿐입니다.")

    # ---- 8. 임계값 변화 실험 ----
    st.header("🎚️ 임계값을 바꾸면 지표는 어떻게 달라질까?")
    rows = []
    for t in np.arange(0.05, 0.96, 0.05):
        p = np.where(proba >= t, positive, classes[0])
        rows.append({
            "임계값": round(t, 2),
            "정확도": accuracy_score(y_test, p),
            "재현율": recall_score(y_test, p, pos_label=positive, zero_division=0),
            "정밀도": precision_score(y_test, p, pos_label=positive, zero_division=0),
        })
    th_df = pd.DataFrame(rows).melt(id_vars="임계값", var_name="지표", value_name="점수")
    fig_th = px.line(th_df, x="임계값", y="점수", color="지표", markers=True, title="임계값별 지표 변화")
    st.plotly_chart(fig_th, use_container_width=True)

    st.warning(
        "이 결과는 한 번의 분할로 얻은 값입니다. random_state를 바꾸거나 교차검증으로 안정성을 확인해야 하고, "
        "이 모델은 학습용 예시이며 실제 진단에 사용할 수 없습니다."
    )

    st.header("💡 더 탐구해 볼 질문")
    st.markdown("""
- 임계값을 낮추면 재현율과 정밀도는 각각 어떻게 변할까? 어떤 값이 적절할까?
- `class_weight`를 해제하면 정확도·재현율·계수는 어떻게 달라질까?
- 규제 강도 `C`를 0.01과 100으로 바꾸면 계수의 크기는 어떻게 변할까?
- 계수가 가장 큰 특성(예: 나이)이 '원인'이라고 말할 수 있을까?
""")


# =============================================================
# 메인: 페이지 선택
# =============================================================
page = st.sidebar.radio(
    "📂 분석 페이지 선택",
    ["🚸 어린이보호구역 교통안전", "📈 Stroke 로지스틱 회귀"],
)
st.sidebar.divider()

if page.startswith("🚸"):
    run_traffic_app()
else:
    run_stroke_app()
