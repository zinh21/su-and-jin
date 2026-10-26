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
YEAR_RE = r"^\s*((?:19|20)\d{2})"


# -----------------------------
# 기본 함수
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


def is_text(series):
    """pandas 버전(object / str)과 상관없이 문자열 열인지 판단"""
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


# -----------------------------
# 어떤 모양의 표든 세로형으로 통일
# 결과 열: __연도, __항목, __값 + 나머지 구분 열
# -----------------------------
def to_long(df):
    df = df.copy()
    df.columns = make_unique(df.columns)

    year_cols = [c for c in df.columns if re.match(YEAR_RE, c)]

    # (A) 연도가 열 이름인 가로형
    if len(year_cols) >= 1:
        items = {c: clean_item(c) for c in year_cols}

        # 첫 행에 항목 이름(사고건수 등)이 들어 있는 경우
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
        return long.dropna(subset=["__값"])

    # (B) 연도가 행(값)으로 들어 있는 세로형
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
    long = pd.concat(parts, ignore_index=True).dropna(subset=["__연도", "__값"])
    long["__연도"] = long["__연도"].astype(int)
    return long


def keep_total_rows(df):
    """'합계', '전국' 같은 총계 행이 있으면 그 행만 남겨 중복 합산을 막는다."""
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
    """항목 이름(또는 구분 열의 값)에서 키워드가 들어간 자료를 찾아 연도별 합계를 만든다."""
    if long is None:
        return None
    norm = lambda s: str(s).replace(" ", "")

    # ① 항목 이름에서 찾기
    for kw in keywords:
        for item in long["__항목"].unique():
            n = norm(item)
            if kw in n and not any(e in n for e in exclude_kw):
                s = yearly(keep_total_rows(long[long["__항목"] == item]))
                if s is not None:
                    return s

    # ② 구분 열의 값에서 찾기
    for kw in keywords:
        for c in long.columns:
            if c.startswith("__") or not is_text(long[c]):
                continue
            vals = long[c].map(norm)
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
            vals = df[c].dropna().astype(str).unique()[:15]
            st.write(f"`{c}` 열의 값 예시:", list(vals))


# -----------------------------
# 데이터 불러오기
# -----------------------------
zone_raw = load_csv(ZONE_URL)
traffic_raw = load_csv(TRAFFIC_URL)

if zone_raw is None or traffic_raw is None:
    st.error("CSV 파일을 불러오지 못했습니다. 인터넷 연결과 주소(공개 저장소인지)를 확인해 주세요.")
    st.stop()

zone_long = to_long(zone_raw)
traffic_long = to_long(traffic_raw)

# -----------------------------
# 어린이보호구역 지정 수
# -----------------------------
zone_series = pick(zone_long, ["지정", "개소", "보호구역"])
if zone_series is None and zone_long is not None and not zone_long.empty:
    first_item = zone_long["__항목"].iloc[0]
    zone_series = yearly(keep_total_rows(zone_long[zone_long["__항목"] == first_item]))

if zone_series is None or len(zone_series) == 0:
    st.error("어린이보호구역 CSV에서 연도/지정 수를 찾지 못했습니다. 아래 구조를 알려주세요.")
    show_structure("어린이보호구역 CSV", zone_raw)
    st.stop()

zones = zone_series.rename("지정 수").reset_index().rename(columns={"__연도": "연도"})

# -----------------------------
# 교통사고 (사고 건수 / 사망자 수 / 부상자 수)
# -----------------------------
death = pick(traffic_long, ["사망"])
injury = pick(traffic_long, ["부상"])
acc = pick(traffic_long, ["사고건수", "발생건수", "사고", "건수"], exclude_kw=["사망", "부상"])

found = {}
for name, s in [("사고 건수", acc), ("사망자 수", death), ("부상자 수", injury)]:
    if s is not None and len(s) > 0:
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
if len(common) >= 1:
    zones = zones[zones["연도"].isin(common)].reset_index(drop=True)
    traffic = traffic[traffic["연도"].isin(common)].reset_index(drop=True)

first_y, last_y = int(traffic["연도"].iloc[0]), int(traffic["연도"].iloc[-1])
st.caption(f"분석 기간: {first_y}년 ~ {last_y}년")

with st.expander("🔧 원본 CSV 구조 확인 (문제가 있을 때 열어 보세요)"):
    show_structure("어린이보호구역 CSV", zone_raw)
    show_structure("교통사고 CSV", traffic_raw)

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
        label = "보호구역 지정 수" if n == "지정 수" else n
        idx[label] = merged[n] / base * 100 if base else None
    idx_long = idx.melt(id_vars="연도", var_name="항목", value_name="지수")
    fig_idx = px.line(
        idx_long, x="연도", y="지수", color="항목", markers=True,
        title=f"{int(merged['연도'].iloc[0])}년 = 100 기준 변화",
    )
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
