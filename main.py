import csv
import re
from io import BytesIO, StringIO

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
MISSING_MARKS = {"", "-", "–", "—", "…", "N/A", "n/a", "NA", "nan", "NaN", "None", "해당없음", "없음"}
FFILL_NAME_HINTS = ["시도", "연도", "구분", "항목", "분류", "지역"]   # 병합 셀로 자주 쓰이는 열 (시군구는 제외)

clean_log = []


# -----------------------------
# 기본 함수
# -----------------------------
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


# -----------------------------
# CSV 읽기 + 파일 구조 정리
# -----------------------------
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_bytes(url):
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        return res.content
    except Exception:
        return None


def read_plain(content):
    """정리 전(원본 그대로) 읽기 - 비교용"""
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(BytesIO(content), encoding=enc)
        except Exception:
            continue
    return None


def fix_structure(content):
    """제목 줄, 병합 셀, 들쭉날쭉한 열 개수, '-' 표기, 빈 행·열을 정리한다."""
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

    # 1) 줄마다 열 개수가 달라도 가장 긴 줄에 맞춰 채움
    width = max(len(r) for r in rows)
    rows = [[c.strip() for c in r] + [""] * (width - len(r)) for r in rows]
    raw = pd.DataFrame(rows).replace(list(MISSING_MARKS), pd.NA)

    info = {"원본 줄 수": len(raw)}

    # 2) 완전히 빈 행·열 삭제
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)

    # 3) 진짜 열 이름 줄 찾기 (앞 10줄 중 값이 충분히 채워진 첫 줄)
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

    # 4) 병합 셀 채우기: 시도/연도/구분 같은 열만 위 값으로 채움
    filled = []
    text_cols = [c for c in df.columns if to_number(df[c].dropna()).notna().mean() < 0.5 if df[c].notna().any()]
    for i, c in enumerate(text_cols):
        if i == 0 or any(h in c for h in FFILL_NAME_HINTS):
            before = int(df[c].isna().sum())
            df[c] = df[c].ffill()
            if before - int(df[c].isna().sum()) > 0:
                filled.append(c)
    info["병합 셀로 보고 채운 열"] = ", ".join(filled) if filled else "없음"

    # 5) 숫자 열은 숫자형으로 변환
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
    long = long.dropna(subset=["__연도", "__값"]).copy()
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


def is_total_row(df):
    mask = pd.Series(False, index=df.index)
    for c in df.columns:
        if c.startswith("__") or not is_text(df[c]):
            continue
        mask |= df[c].astype(str).str.strip().isin(TOTAL_WORDS)
    return mask


def remove_outliers(long, name):
    if long is None or long.empty:
        return long
    total_mask = is_total_row(long)
    keep = pd.Series(True, index=long.index)
    for item, g in long.groupby("__항목"):
        detail = g[~total_mask.loc[g.index]]
        if len(detail) < 4:
            continue
        q1, q3 = detail["__값"].quantile(0.25), detail["__값"].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        keep.loc[detail.index[(detail["__값"] < lower) | (detail["__값"] > upper)]] = False
    before = len(long)
    cleaned = long[keep].copy()
    clean_log.append({
        "자료": name, "단계": "이상치 처리",
        "처리 방법": "항목별 IQR(Q1-1.5×IQR ~ Q3+1.5×IQR) 밖의 세부 행 삭제, 총계 행 보존",
        "처리 전 행 수": before, "삭제된 행 수": before - len(cleaned), "처리 후 행 수": len(cleaned),
    })
    return cleaned


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


# -----------------------------
# 데이터 불러오기 + 구조 정리
# -----------------------------
zone_bytes = load_bytes(ZONE_URL)
traffic_bytes = load_bytes(TRAFFIC_URL)

if zone_bytes is None or traffic_bytes is None:
    st.error("CSV 파일을 불러오지 못했습니다. 인터넷 연결과 주소(공개 저장소인지)를 확인해 주세요.")
    st.stop()

zone_plain, traffic_plain = read_plain(zone_bytes), read_plain(traffic_bytes)
zone_raw, zone_info = fix_structure(zone_bytes)
traffic_raw, traffic_info = fix_structure(traffic_bytes)

if zone_raw is None or traffic_raw is None:
    st.error("CSV 구조를 정리하지 못했습니다. 파일을 확인해 주세요.")
    st.stop()

# -----------------------------
# 데이터 개요 (구조 정리 전/후 비교)
# -----------------------------
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

# -----------------------------
# 전처리 (결측치·이상치)
# -----------------------------
zone_long = remove_outliers(to_long(zone_raw, "어린이보호구역"), "어린이보호구역")
traffic_long = remove_outliers(to_long(traffic_raw, "교통사고"), "교통사고")

st.header("🧹 결측치·이상치 처리 결과")
if clean_log:
    st.dataframe(pd.DataFrame(clean_log), hide_index=True, use_container_width=True)
st.caption(
    "결측치는 숫자가 아니거나 비어 있는 값의 행을 삭제했습니다. "
    "이상치는 항목별 IQR 범위를 벗어난 세부 행을 삭제했고, 합계·전국 같은 총계 행은 보존했습니다."
)
st.divider()

# -----------------------------
# 어린이보호구역 지정 수
# -----------------------------
zone_series = pick(zone_long, ["지정", "개소", "보호구역"])
if zone_series is None and zone_long is not None and not zone_long.empty:
    first_item = zone_long["__항목"].iloc[0]
    zone_series = yearly(keep_total_rows(zone_long[zone_long["__항목"] == first_item]))

if zone_series is None or len(zone_series) == 0:
    st.error("어린이보호구역 CSV에서 연도/지정 수를 찾지 못했습니다. 아래 구조를 알려주세요.")
    show_structure("어린이보호구역 CSV (정리 후)", zone_raw)
    st.stop()

zones = zone_series.rename("지정 수").reset_index().rename(columns={"__연도": "연도"})

# -----------------------------
# 교통사고
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
    show_structure("교통사고 CSV (정리 후)", traffic_raw)
    st.stop()

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
    delta = f"{last - traffic[name].iloc[-2]:+,.0f} (전년 대비)" if len(traffic) > 1 else None
    cols[i + 1].metric(f"{last_y}년 {name}", f"{last:,.0f}", delta=delta, delta_color="inverse")

st.divider()

# -----------------------------
# 그래프 + 발견 문장
# -----------------------------
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
- 구조를 정리하기 전과 후에 빈 값 개수는 얼마나 달라졌을까?
- 남은 빈 값은 '0건'일까, '자료 없음'일까?
- 이상치로 삭제된 행은 정말 잘못된 값일까, 실제로 중요한 사건일까?
- 사고 건수뿐 아니라 차량 통행량을 고려하면 결론이 달라질까?
""")

st.caption("자료: 제공된 어린이보호구역 및 교통사고통계 CSV 파일")
