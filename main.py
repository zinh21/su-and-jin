import re
from io import BytesIO

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

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

TOTAL_WORDS = ["합계", "전국", "계", "전체", "총계"]


# -----------------------------
# 함수
# -----------------------------
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_csv(url):
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
    except Exception:
        return None
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(BytesIO(res.content), encoding=enc)
        except Exception:
            continue
    return None


def to_number(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def to_year(series):
    return pd.to_numeric(series.astype(str).str.extract(r"((?:19|20)\d{2})")[0], errors="coerce")


def pct(a, b):
    return (a - b) / b * 100 if b else 0


def find_year_col(df):
    for c in df.columns:
        if any(k in str(c) for k in ["연도", "년도", "기간", "년", "year", "Year"]):
            if to_year(df[c]).notna().mean() > 0.5:
                return c
    for c in df.columns:
        if to_year(df[c]).notna().mean() > 0.8:
            return c
    return None


def make_unique(columns):
    """중복된 열 이름 뒤에 _1, _2를 붙여 고유하게 만든다."""
    seen = {}
    result = []
    for c in columns:
        c = str(c).strip()
        if c in seen:
            seen[c] += 1
            result.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            result.append(c)
    return result


def prepare(df):
    """열 이름 정리 + '2022년 사고건수'처럼 연도가 열 이름에 있으면 세로형으로 변환"""
    df = df.copy()
    df.columns = make_unique(df.columns)

    year_cols = [c for c in df.columns if re.match(r"^(19|20)\d{2}", c)]
    if len(year_cols) < 2:
        return df

    id_cols = [c for c in df.columns if c not in year_cols]
    long = df.melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name="__열이름__",
        value_name="__값__",
        ignore_index=True,
    )

    # '2022년 사고건수' -> 연도=2022, 항목='사고건수'
    parts = long["__열이름__"].str.extract(r"^((?:19|20)\d{2})\D*(.*)$")
    long["__연도__"] = parts[0]
    long["__항목__"] = parts[1].str.replace(r"_\d+$", "", regex=True).str.strip()

    # 기존 열과 이름이 겹치지 않도록 정리
    long = long.drop(columns=["__열이름__"])
    for old in ["연도", "값"]:
        if old in long.columns:
            long = long.rename(columns={old: f"{old}_원본"})
    long = long.rename(columns={"__연도__": "연도", "__값__": "값"})

    # 항목 이름이 있으면 구분 열로 사용 (사고건수/사망자수/부상자수 등)
    if (long["__항목__"] != "").any():
        if "구분" in long.columns:
            long = long.rename(columns={"구분": "구분_원본"})
        long = long.rename(columns={"__항목__": "구분"})
    else:
        long = long.drop(columns=["__항목__"])

    return long


def keep_total_rows(df, skip):
    """'합계', '전국' 같은 총계 행이 있으면 그 행만 남겨 중복 합산 방지"""
    for c in df.columns:
        if c in skip or df[c].dtype != object:
            continue
        mask = df[c].astype(str).str.strip().isin(TOTAL_WORDS)
        if mask.any():
            df = df[mask]
    return df


def numeric_cols(df, exclude):
    return [c for c in df.columns if c not in exclude and to_number(df[c]).notna().mean() > 0.5]


def by_year(df, ycol, vcol):
    out = pd.DataFrame({"연도": to_year(df[ycol]), "값": to_number(df[vcol])}).dropna()
    if out.empty:
        return None
    out["연도"] = out["연도"].astype(int)
    return out.groupby("연도")["값"].sum().sort_index()


def extract(df, ycol, keywords, exclude_kw=()):
    """① 열 이름에서 찾고, ② 없으면 행의 값(구분 열 등)에서 찾는다."""
    # ① 가로형: 열 이름에 키워드
    for kw in keywords:
        for c in df.columns:
            name = c.replace(" ", "")
            if c == ycol or kw not in name or any(e in name for e in exclude_kw):
                continue
            if to_number(df[c]).notna().mean() > 0.5:
                sub = keep_total_rows(df, [ycol, c])
                s = by_year(sub, ycol, c)
                if s is not None:
                    return s
    # ② 세로형: 행의 값에 키워드
    pattern = "|".join(keywords)
    for c in df.columns:
        if c == ycol or df[c].dtype != object:
            continue
        vals = df[c].astype(str).str.replace(" ", "", regex=False)
        mask = vals.str.contains(pattern, regex=True)
        if exclude_kw:
            mask &= ~vals.str.contains("|".join(exclude_kw), regex=True)
        if not mask.any():
            continue
        sub = df[mask]
        vcol = "값" if "값" in sub.columns else None
        if vcol is None:
            nums = numeric_cols(sub, [ycol, c])
            vcol = nums[0] if nums else None
        if vcol is None:
            continue
        sub = keep_total_rows(sub, [ycol, c, vcol])
        s = by_year(sub, ycol, vcol)
        if s is not None:
            return s
    return None


def show_structure(title, df):
    st.write(f"**{title}** 열 이름:", list(df.columns))
    st.dataframe(df.head(10), use_container_width=True)


# -----------------------------
# 데이터 불러오기
# -----------------------------
zone_raw = load_csv(ZONE_URL)
traffic_raw = load_csv(TRAFFIC_URL)

if zone_raw is None or traffic_raw is None:
    st.error("CSV 파일을 불러오지 못했습니다. 인터넷 연결과 주소(공개 저장소인지)를 확인해 주세요.")
    st.stop()

zone_df = prepare(zone_raw)
traffic_df = prepare(traffic_raw)

# -----------------------------
# 어린이보호구역 데이터
# -----------------------------
z_year = find_year_col(zone_df)
zone_series = None
if z_year is not None:
    if "값" in zone_df.columns:
        sub = keep_total_rows(zone_df, [z_year, "값"])
        zone_series = by_year(sub, z_year, "값")
    else:
        zone_series = extract(zone_df, z_year, ["지정", "개소", "보호구역", "합계"])
        if zone_series is None:
            nums = numeric_cols(zone_df, [z_year])
            if nums:
                sub = keep_total_rows(zone_df, [z_year, nums[-1]])
                zone_series = by_year(sub, z_year, nums[-1])

if zone_series is None or len(zone_series) == 0:
    st.error("어린이보호구역 CSV에서 연도/지정 수를 찾지 못했습니다. 아래 구조를 알려주세요.")
    show_structure("어린이보호구역 CSV", zone_raw)
    st.stop()

zones = zone_series.rename("지정 수").reset_index()

# -----------------------------
# 교통사고 데이터
# -----------------------------
t_year = find_year_col(traffic_df)
if t_year is None:
    st.error("교통사고 CSV에서 연도를 찾지 못했습니다. 아래 구조를 알려주세요.")
    show_structure("교통사고 CSV", traffic_raw)
    st.stop()

found = {}
death = extract(traffic_df, t_year, ["사망"])
injury = extract(traffic_df, t_year, ["부상"])
acc = extract(traffic_df, t_year, ["사고건수", "발생건수", "사고", "건수"], exclude_kw=["사망", "부상"])
for name, s in [("사고 건수", acc), ("사망자 수", death), ("부상자 수", injury)]:
    if s is not None:
        found[name] = s

if not found:
    st.error("교통사고 CSV에서 사고/사망/부상 항목을 찾지 못했습니다. 아래 구조를 알려주세요.")
    show_structure("교통사고 CSV", traffic_raw)
    st.stop()

traffic = pd.DataFrame(found).sort_index().reset_index()
traffic = traffic.rename(columns={traffic.columns[0]: "연도"})
items = list(found.keys())

# 두 자료에서 겹치는 연도만 사용
common = sorted(set(zones["연도"]) & set(traffic["연도"]))
if len(common) >= 2:
    zones = zones[zones["연도"].isin(common)].reset_index(drop=True)
    traffic = traffic[traffic["연도"].isin(common)].reset_index(drop=True)

first_y, last_y = int(traffic["연도"].iloc[0]), int(traffic["연도"].iloc[-1])
st.caption(f"분석 기간: {first_y}년 ~ {last_y}년")

# -----------------------------
# 핵심 수치
# -----------------------------
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
    delta = None
    if len(traffic) > 1:
        delta = f"{last - traffic[name].iloc[-2]:+,.0f} (전년 대비)"
    cols[i + 1].metric(f"{last_y}년 {name}", f"{last:,.0f}", delta=delta, delta_color="inverse")

st.divider()

# -----------------------------
# 그래프
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

# 변화율 비교
st.subheader("📈 시작 연도를 100으로 놓고 본 변화율 비교")
merged = pd.merge(zones, traffic, on="연도", how="inner")
if len(merged) >= 2:
    idx = pd.DataFrame({"연도": merged["연도"]})
    for n in ["지정 수"] + items:
        base = merged[n].iloc[0]
        idx["보호구역 지정 수" if n == "지정 수" else n] = merged[n] / base * 100 if base else None
    long = idx.melt(id_vars="연도", var_name="항목", value_name="지수")
    fig_idx = px.line(long, x="연도", y="지수", color="항목", markers=True,
                      title=f"{int(merged['연도'].iloc[0])}년 = 100 기준 변화")
    fig_idx.update_layout(xaxis=dict(type="category"))
    st.plotly_chart(fig_idx, use_container_width=True)
    st.caption("단위가 다른 항목도 같은 기준(시작 연도 = 100)으로 바꾸면 변화 방향을 비교하기 쉬워요.")
else:
    st.info("두 자료에 겹치는 연도가 2개 미만이라 변화율 비교는 생략했습니다.")

st.divider()

# -----------------------------
# 데이터 표
# -----------------------------
st.header("📋 정리된 데이터 표")
st.dataframe(merged if not merged.empty else traffic, hide_index=True, use_container_width=True)

st.divider()

# -----------------------------
# 자동 분석
# -----------------------------
st.header("🔍 데이터에서 발견한 점")

diff = z_last - z_first
trend = "증가" if diff > 0 else "감소" if diff < 0 else "변화 없음"
st.subheader("1. 보호구역 지정 수")
st.write(
    f"어린이보호구역 지정 수는 {int(zones['연도'].iloc[0])}년 {z_first:,.0f}개에서 "
    f"{int(zones['연도'].iloc[-1])}년 {z_last:,.0f}개로 "
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
