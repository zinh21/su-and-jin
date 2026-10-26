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
st.markdown("어린이보호구역 지정 현황과 교통사고 변화를 한눈에 살펴봅니다.")

ZONE_URL = "https://raw.githubusercontent.com/zinh21/su-and-jin/refs/heads/main/161601_20261026083839280_excel.csv"
TRAFFIC_URL = "https://raw.githubusercontent.com/zinh21/su-and-jin/refs/heads/main/%EA%B5%90%ED%86%B5%EC%82%AC%EA%B3%A0%ED%86%B5%EA%B3%84_20261026.csv"


# -----------------------------
# 함수
# -----------------------------
@st.cache_data
def load_csv(url):
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(url, encoding=enc)
        except Exception:
            continue
    return None


def to_number(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def to_year(series):
    return pd.to_numeric(series.astype(str).str.extract(r"(\d{4})")[0], errors="coerce")


def find_year_col(df):
    for c in df.columns:
        if any(k in str(c) for k in ["연도", "년도", "기간", "year", "Year"]):
            return c
    for c in df.columns:
        y = to_year(df[c])
        if y.between(1990, 2100).mean() > 0.8:
            return c
    return None


def wide_to_long(df):
    """2022, 2023처럼 연도가 열 이름인 표를 세로형으로 바꿉니다."""
    year_cols = [c for c in df.columns if str(c).strip()[:4].isdigit() and len(str(c).strip()) <= 6]
    if not year_cols:
        return None
    id_cols = [c for c in df.columns if c not in year_cols]
    long = df.melt(id_vars=id_cols, value_vars=year_cols, var_name="연도", value_name="값")
    return long


def keep_total_rows(df, skip_cols):
    """'합계', '전국' 같은 총계 행이 있으면 그 행만 남겨 중복 합산을 막습니다."""
    for c in df.columns:
        if c in skip_cols or df[c].dtype != object:
            continue
        vals = df[c].astype(str).str.strip()
        mask = vals.isin(["합계", "전국", "계", "전체", "총계"])
        if mask.any():
            df = df[mask]
    return df


def find_value_col(df, keywords, exclude=()):
    for kw in keywords:
        for c in df.columns:
            if c in exclude:
                continue
            if kw in str(c).replace(" ", "") and to_number(df[c]).notna().mean() > 0.5:
                return c
    return None


def pct(a, b):
    return (a - b) / b * 100 if b else 0


# -----------------------------
# 데이터 불러오기
# -----------------------------
zone_raw = load_csv(ZONE_URL)
traffic_raw = load_csv(TRAFFIC_URL)

if zone_raw is None or traffic_raw is None:
    st.error("CSV 파일을 불러오지 못했습니다. 주소와 인터넷 연결을 확인해 주세요.")
    st.stop()

# -----------------------------
# 어린이보호구역 데이터 정리
# -----------------------------
zone_year = find_year_col(zone_raw)
if zone_year is not None:
    zone_df = zone_raw.copy()
    zone_val = find_value_col(zone_df, ["지정", "개소", "구역", "수", "합계"], exclude=[zone_year])
    if zone_val is None:
        nums = [c for c in zone_df.columns if c != zone_year and to_number(zone_df[c]).notna().mean() > 0.5]
        zone_val = nums[-1] if nums else None
else:
    zone_df = wide_to_long(zone_raw)
    zone_year, zone_val = "연도", "값"

if zone_df is None or zone_val is None:
    st.error("어린이보호구역 CSV에서 연도와 지정 수 열을 찾지 못했습니다. 아래 열 이름을 알려주세요.")
    st.write(list(zone_raw.columns))
    st.stop()

zone_df = keep_total_rows(zone_df, [zone_year, zone_val])
zones = pd.DataFrame({
    "연도": to_year(zone_df[zone_year]),
    "지정 수": to_number(zone_df[zone_val]),
}).dropna()
zones["연도"] = zones["연도"].astype(int)
zones = zones.groupby("연도", as_index=False).sum().sort_values("연도")

# -----------------------------
# 교통사고 데이터 정리
# -----------------------------
t_year = find_year_col(traffic_raw)
if t_year is None:
    st.error("교통사고 CSV에서 연도 열을 찾지 못했습니다. 아래 열 이름을 알려주세요.")
    st.write(list(traffic_raw.columns))
    st.stop()

t_df = keep_total_rows(traffic_raw, [t_year])
death_col = find_value_col(t_df, ["사망"], exclude=[t_year])
injury_col = find_value_col(t_df, ["부상"], exclude=[t_year])
acc_col = find_value_col(t_df, ["사고건수", "발생건수", "건수", "사고"], exclude=[t_year, death_col, injury_col])

found = {"사고 건수": acc_col, "사망자 수": death_col, "부상자 수": injury_col}
found = {k: v for k, v in found.items() if v is not None}

if not found:
    st.error("교통사고 CSV에서 사고/사망/부상 열을 찾지 못했습니다. 아래 열 이름을 알려주세요.")
    st.write(list(traffic_raw.columns))
    st.stop()

traffic = pd.DataFrame({"연도": to_year(t_df[t_year])})
for name, col in found.items():
    traffic[name] = to_number(t_df[col])
traffic = traffic.dropna(subset=["연도"])
traffic["연도"] = traffic["연도"].astype(int)
traffic = traffic.groupby("연도", as_index=False).sum().sort_values("연도")
items = list(found.keys())

# 두 자료에서 겹치는 연도만 사용
common = sorted(set(zones["연도"]) & set(traffic["연도"]))
if len(common) >= 2:
    zones = zones[zones["연도"].isin(common)]
    traffic = traffic[traffic["연도"].isin(common)]

first_y, last_y = int(zones["연도"].iloc[0]), int(zones["연도"].iloc[-1])
st.caption(f"분석 기간: {first_y}년 ~ {last_y}년")

# -----------------------------
# 핵심 수치
# -----------------------------
st.header("📌 핵심 데이터")
cols = st.columns(1 + len(items))

z_last, z_first = zones["지정 수"].iloc[-1], zones["지정 수"].iloc[0]
cols[0].metric(
    f"{last_y}년 보호구역 지정 수",
    f"{z_last:,.0f}개",
    delta=f"{z_last - z_first:+,.0f}개 ({first_y}년 대비)",
)
for i, name in enumerate(items):
    last = traffic[name].iloc[-1]
    delta = None
    if len(traffic) > 1:
        delta = f"{last - traffic[name].iloc[-2]:+,.0f} (전년 대비)"
    # 사고·사망·부상은 늘면 나쁜 것이므로 색을 반대로 표시
    cols[i + 1].metric(f"{last_y}년 {name}", f"{last:,.0f}", delta=delta, delta_color="inverse")

st.divider()

# -----------------------------
# 그래프 (한 화면에 모두 표시)
# -----------------------------
st.header("📊 한눈에 보는 그래프")

def bar(df, col, title, color):
    fig = px.bar(df, x="연도", y=col, text=col, title=title, color_discrete_sequence=[color])
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(xaxis=dict(type="category"), yaxis_title=None)
    return fig

colors = {"사고 건수": "#EF553B", "사망자 수": "#636EFA", "부상자 수": "#FFA15A"}
charts = [("지정 수", zones, "연도별 어린이보호구역 지정 수", "#00CC96")]
charts += [(n, traffic, f"연도별 {n}", colors[n]) for n in items]

for i in range(0, len(charts), 2):
    row = st.columns(2)
    for j, (col, df, title, color) in enumerate(charts[i:i + 2]):
        row[j].plotly_chart(bar(df, col, title, color), use_container_width=True)

# 변화 추이 비교 (2025 기준 100으로 환산 → 시작 연도 = 100)
st.subheader("📈 시작 연도를 100으로 놓고 본 변화율 비교")
index_df = pd.DataFrame({"연도": zones["연도"].values})
index_df["보호구역 지정 수"] = zones["지정 수"].values / zones["지정 수"].iloc[0] * 100
tr = traffic.reset_index(drop=True)
for n in items:
    base = tr[n].iloc[0]
    index_df[n] = tr[n] / base * 100 if base else None

long = index_df.melt(id_vars="연도", var_name="항목", value_name="지수")
fig_idx = px.line(long, x="연도", y="지수", color="항목", markers=True,
                  title=f"{first_y}년 = 100 기준 변화")
fig_idx.update_layout(xaxis=dict(type="category"))
st.plotly_chart(fig_idx, use_container_width=True)
st.caption("단위가 서로 다른 항목도 같은 기준(시작 연도 = 100)으로 바꾸면 변화 방향을 비교하기 쉬워요.")

st.divider()

# -----------------------------
# 데이터 표
# -----------------------------
st.header("📋 정리된 데이터 표")
merged = pd.merge(zones, traffic, on="연도", how="inner")
st.dataframe(merged, hide_index=True, use_container_width=True)

st.divider()

# -----------------------------
# 자동 분석
# -----------------------------
st.header("🔍 데이터에서 발견한 점")

diff = z_last - z_first
trend = "증가" if diff > 0 else "감소" if diff < 0 else "변화 없음"
st.subheader("1. 보호구역 지정 수")
st.write(
    f"어린이보호구역 지정 수는 {first_y}년 {z_first:,.0f}개에서 {last_y}년 {z_last:,.0f}개로 "
    f"{abs(diff):,.0f}개({abs(pct(z_last, z_first)):.1f}%) **{trend}**했습니다."
)

for k, name in enumerate(items, start=2):
    s = traffic[name].reset_index(drop=True)
    yrs = traffic["연도"].reset_index(drop=True)
    peak_y, low_y = int(yrs[s.idxmax()]), int(yrs[s.idxmin()])
    d = s.iloc[-1] - s.iloc[0]
    t = "증가" if d > 0 else "감소" if d < 0 else "변화 없음"
    st.subheader(f"{k}. {name}")
    line = " → ".join(f"{int(y)}년 {v:,.0f}" for y, v in zip(yrs, s))
    st.write(
        f"{line}\n\n"
        f"{first_y}년 대비 {last_y}년에는 **{t}**({d:+,.0f}, {pct(s.iloc[-1], s.iloc[0]):+.1f}%)했고, "
        f"가장 많았던 해는 **{peak_y}년**, 가장 적었던 해는 **{low_y}년**입니다."
    )

st.warning(
    "주의: 이 데이터만으로 보호구역 지정 수의 변화가 교통사고 변화의 원인이라고 단정할 수 없습니다. "
    "차량 통행량, 보호구역별 사고율, 안전시설 변화 등의 자료를 함께 분석해야 합니다."
)

st.divider()

st.header("💡 더 탐구해 볼 질문")
st.markdown("""
- 지정 수가 변한 이유는 무엇일까? (학령인구 감소, 학교 통폐합 등)
- 사고 건수뿐 아니라 차량 통행량을 고려하면 결론이 달라질까?
- 사망자 수처럼 아주 작은 숫자의 변화는 어떻게 해석해야 할까?
- 어린이 교통사고를 줄이기 위해 어떤 정책이 필요할까?
""")

st.caption("자료: 제공된 어린이보호구역 및 교통사고통계 CSV 파일")
