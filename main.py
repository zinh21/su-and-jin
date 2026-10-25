import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="어린이보호구역 교통안전 분석",
    page_icon="🚸",
    layout="wide",
)

st.title("🚸 어린이보호구역 교통안전 분석")
st.markdown("""
CSV 자료를 이용해 어린이보호구역 지정 현황과 교통사고 변화를 분석합니다.
왼쪽 사이드바에서 분석에 사용할 **열(컬럼)** 을 선택해 보세요.
""")

# -----------------------------
# 데이터 주소
# -----------------------------
ZONE_URL = "https://raw.githubusercontent.com/zinh21/su-and-jin/refs/heads/main/161601_20261026083839280_excel.csv"
TRAFFIC_URL = "https://raw.githubusercontent.com/zinh21/su-and-jin/refs/heads/main/%EA%B5%90%ED%86%B5%EC%82%AC%EA%B3%A0%ED%86%B5%EA%B3%84_20261026.csv"


# -----------------------------
# 함수
# -----------------------------
@st.cache_data
def load_csv(url):
    """인코딩을 바꿔가며 CSV를 읽어옵니다."""
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(url, encoding=enc)
        except Exception:
            continue
    return None


def to_number(series):
    """쉼표 등이 섞인 열을 숫자로 변환합니다."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def to_year(series):
    """'2022', '2022년' 같은 값에서 4자리 연도를 추출합니다."""
    return pd.to_numeric(series.astype(str).str.extract(r"(\d{4})")[0], errors="coerce")


def apply_filter(df, key, label):
    """원하는 열의 값으로 데이터를 걸러냅니다(선택 사항)."""
    filter_col = st.sidebar.selectbox(
        f"{label} 필터 열 (선택)", ["(사용 안 함)"] + list(df.columns), key=f"{key}_fcol"
    )
    if filter_col != "(사용 안 함)":
        options = df[filter_col].dropna().unique().tolist()
        chosen = st.sidebar.multiselect(
            f"{label} - '{filter_col}' 값 선택", options, default=options, key=f"{key}_fval"
        )
        df = df[df[filter_col].isin(chosen)]
    return df


# -----------------------------
# 데이터 불러오기
# -----------------------------
zone_raw = load_csv(ZONE_URL)
traffic_raw = load_csv(TRAFFIC_URL)

if zone_raw is None or traffic_raw is None:
    st.error("CSV 파일을 불러오지 못했습니다. 주소와 인터넷 연결을 확인해 주세요.")
    st.stop()

st.header("📂 원본 데이터 확인")
with st.expander("어린이보호구역 CSV 보기"):
    st.write(f"행 {zone_raw.shape[0]}개 × 열 {zone_raw.shape[1]}개")
    st.dataframe(zone_raw, use_container_width=True)

with st.expander("교통사고통계 CSV 보기"):
    st.write(f"행 {traffic_raw.shape[0]}개 × 열 {traffic_raw.shape[1]}개")
    st.dataframe(traffic_raw, use_container_width=True)

# -----------------------------
# 사이드바: 열 선택
# -----------------------------
st.sidebar.header("⚙️ 어린이보호구역 CSV 설정")
zone_df = apply_filter(zone_raw, "zone", "보호구역")
zone_year_col = st.sidebar.selectbox("연도 열", zone_df.columns, key="zone_year")
zone_val_col = st.sidebar.selectbox(
    "지정 수(값) 열", zone_df.columns, index=min(1, len(zone_df.columns) - 1), key="zone_val"
)

st.sidebar.header("⚙️ 교통사고통계 CSV 설정")
traffic_df = apply_filter(traffic_raw, "traffic", "교통사고")
traffic_year_col = st.sidebar.selectbox("연도 열 ", traffic_df.columns, key="traffic_year")
traffic_val_cols = st.sidebar.multiselect(
    "분석할 값 열 (사고 건수, 사망자 수, 부상자 수 등)",
    [c for c in traffic_df.columns if c != traffic_year_col],
    default=[],
    key="traffic_vals",
)

# -----------------------------
# 데이터 정리
# -----------------------------
zones = pd.DataFrame({
    "연도": to_year(zone_df[zone_year_col]),
    "어린이보호구역 지정 수": to_number(zone_df[zone_val_col]),
}).dropna()
zones["연도"] = zones["연도"].astype(int)
zones = zones.groupby("연도", as_index=False).sum().sort_values("연도")

traffic = pd.DataFrame()
if traffic_val_cols:
    traffic = pd.DataFrame({"연도": to_year(traffic_df[traffic_year_col])})
    for c in traffic_val_cols:
        traffic[c] = to_number(traffic_df[c])
    traffic = traffic.dropna(subset=["연도"])
    traffic["연도"] = traffic["연도"].astype(int)
    traffic = traffic.groupby("연도", as_index=False).sum().sort_values("연도")

if zones.empty:
    st.warning("어린이보호구역 데이터에서 연도/값을 읽지 못했습니다. 사이드바에서 열을 다시 선택해 보세요.")
    st.stop()

if traffic.empty:
    st.info("👈 사이드바에서 교통사고통계의 **연도 열**과 **분석할 값 열**을 선택해 주세요.")

# -----------------------------
# 핵심 수치
# -----------------------------
st.divider()
st.header("📌 핵심 데이터")

latest_zone = zones.iloc[-1]
first_zone = zones.iloc[0]

metric_cols = st.columns(1 + len(traffic_val_cols)) if traffic_val_cols else st.columns(1)

with metric_cols[0]:
    st.metric(
        f"{int(latest_zone['연도'])}년 보호구역 지정 수",
        f"{latest_zone['어린이보호구역 지정 수']:,.0f}개",
        delta=f"{latest_zone['어린이보호구역 지정 수'] - first_zone['어린이보호구역 지정 수']:+,.0f}개 ({int(first_zone['연도'])}년 대비)",
    )

if not traffic.empty:
    for i, c in enumerate(traffic_val_cols):
        with metric_cols[i + 1]:
            latest = traffic.iloc[-1]
            delta = None
            if len(traffic) > 1:
                delta = f"{latest[c] - traffic.iloc[-2][c]:+,.0f} (전년 대비)"
            st.metric(f"{int(latest['연도'])}년 {c}", f"{latest[c]:,.0f}", delta=delta)

st.divider()

# -----------------------------
# 그래프 1: 보호구역 지정 현황
# -----------------------------
st.header("1. 어린이보호구역 지정 현황")

fig1 = px.bar(
    zones,
    x="연도",
    y="어린이보호구역 지정 수",
    text="어린이보호구역 지정 수",
    title="연도별 어린이보호구역 지정 현황",
)
fig1.update_traces(texttemplate="%{text:,}", textposition="outside")
fig1.update_layout(xaxis=dict(type="category"))
st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# 그래프 2~: 교통사고 항목별
# -----------------------------
if not traffic.empty:
    for n, c in enumerate(traffic_val_cols, start=2):
        st.divider()
        st.header(f"{n}. 연도별 {c}")
        fig = px.bar(
            traffic,
            x="연도",
            y=c,
            text=c,
            title=f"연도별 {c}",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(xaxis=dict(type="category"))
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # 비교 그래프
    # -------------------------
    if len(traffic_val_cols) >= 2:
        st.divider()
        st.header(f"{len(traffic_val_cols) + 2}. 항목별 변화 비교")

        traffic_long = traffic.melt(
            id_vars="연도",
            value_vars=traffic_val_cols,
            var_name="항목",
            value_name="값",
        )
        fig_line = px.line(
            traffic_long,
            x="연도",
            y="값",
            color="항목",
            markers=True,
            text="값",
            title="연도별 항목 비교",
        )
        fig_line.update_traces(textposition="top center")
        fig_line.update_layout(xaxis=dict(type="category"))
        st.plotly_chart(fig_line, use_container_width=True)

    # -------------------------
    # 두 자료 합쳐서 비교
    # -------------------------
    st.divider()
    st.header("🔗 보호구역 지정 수와 교통사고 함께 보기")

    merged = pd.merge(zones, traffic, on="연도", how="inner")
    if merged.empty:
        st.info("두 자료에 겹치는 연도가 없어 합칠 수 없습니다.")
    else:
        st.dataframe(merged, hide_index=True, use_container_width=True)

        target = st.selectbox("보호구역 지정 수와 비교할 항목", traffic_val_cols)
        fig_sc = px.scatter(
            merged,
            x="어린이보호구역 지정 수",
            y=target,
            text="연도",
            title=f"보호구역 지정 수와 {target}의 관계",
        )
        fig_sc.update_traces(textposition="top center", marker=dict(size=12))
        st.plotly_chart(fig_sc, use_container_width=True)

        if len(merged) >= 3:
            corr = merged["어린이보호구역 지정 수"].corr(merged[target])
            st.write(f"상관계수: **{corr:.2f}**")
        st.warning(
            "주의: 두 값이 함께 변한다고 해서 한쪽이 다른 쪽의 원인이라고 단정할 수 없습니다. "
            "차량 통행량, 안전시설 변화 등 다른 요인도 함께 생각해 보세요."
        )

# -----------------------------
# 데이터 표
# -----------------------------
st.divider()
st.header("📋 정리된 데이터 표")

with st.expander("어린이보호구역 지정 현황 (연도별 합계)"):
    st.dataframe(zones, hide_index=True, use_container_width=True)

if not traffic.empty:
    with st.expander("교통사고 현황 (연도별 합계)"):
        st.dataframe(traffic, hide_index=True, use_container_width=True)

# -----------------------------
# 탐구 질문
# -----------------------------
st.divider()
st.header("💡 더 탐구해 볼 질문")
st.markdown("""
- 그래프에서 가장 크게 변한 해는 언제이고, 그 이유는 무엇일까?
- 지역(시도)별로 필터를 바꾸면 결과가 어떻게 달라질까?
- 사고 건수뿐 아니라 차량 통행량을 고려하면 결론이 달라질까?
- 어린이 교통사고를 줄이기 위해 어떤 정책이 필요할까?
""")

st.caption("자료: 제공된 어린이보호구역 및 교통사고통계 CSV 파일")
