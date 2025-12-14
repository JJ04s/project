import streamlit as st
import pandas as pd
from collections import defaultdict
import plotly.graph_objects as go
import re
import copy
import folium
from streamlit_folium import st_folium
import numpy as np

def clean_name(x):
    if pd.isna(x):
        return x
    x = re.sub(r"\(.*?\)", "", x)
    x = x.replace("역", "")
    x = x.replace(" ", "")
    return x

# 다익스트라
def dijkstra(graph, start, end):

    path = {node: [] for node in graph}
    path[start] = start
    distance = {node: 1000 for node in graph}
    distance[start] = 0
    visited = set()

    for node in graph[start]:
        distance[node] = graph[start][node]
        path[node] = [start, node]
    
    while end not in visited:
        min_node = None
        min_dist = 1000
        for node in distance:
            if distance[node] < min_dist and node not in visited:
                min_node = node
                min_dist = distance[node]

        if min_node == None:
            return None
        
        visited.add(min_node)

        for node in graph[min_node]:
            if (distance[min_node] + graph[min_node][node]) < distance[node]:
                distance[node] = distance[min_node] + graph[min_node][node]

                path[node] = path[min_node].copy()
                path[node].append(node)

    return path[end], distance[end]

# 역에서 정보를 추출
def get_station_info(merged):
    info = {}
    df = merged.drop_duplicates("역명_cleaned")

    for _, row in df.iterrows():
        name = row["역명_cleaned"]
        info[name] = {
            "line": row["호선"],
            "order": row["stinConsOrdr"],
            "left": row["left_transfer"],
            "left_dist": row["left_dist"],
            "right": row["right_transfer"],
            "right_dist": row["right_dist"],
        }
    
    return info

# 같은 호선의 거리를 탐색
def find_direct_path(start, end, info):
    if info[start]["line"] != info[end]["line"]:
        return None
    
    order_s = info[start]["order"]
    order_e = info[end]["order"]

    return abs(order_s - order_e)

# 역 간의 거리와 경로를 탐색
def find_path(adj_graph, start, end, merged):

    info = get_station_info(merged)
    station_graph = copy.deepcopy(adj_graph)

    if start not in station_graph:
        station_graph[start] = {}

    if info[start]["line"] == info[end]["line"]:
        direct_path = find_direct_path(start, end, info)
        s = {end: direct_path}
        e = {start: direct_path}

    else:
        s = {}
        e = {}

    if start not in adj_graph:
        left_s = info[start]["left"]
        l_dist_s = info[start]["left_dist"]
        right_s = info[start]["right"]
        r_dist_s = info[start]["right_dist"]

        if pd.notna(left_s):
            s[left_s] = l_dist_s
            (station_graph[left_s])[start] = l_dist_s
        if pd.notna(right_s):
            s[right_s] = r_dist_s
            (station_graph[right_s])[start] = r_dist_s

        if len(s) > 0:
            station_graph[start] = s

    if end not in adj_graph:

        left_e = info[end]["left"]
        l_dist_e = info[end]["left_dist"]
        right_e = info[end]["right"]
        r_dist_e = info[end]["right_dist"]

        if pd.notna(left_e):
            e[left_e] = l_dist_e
            (station_graph[left_e])[end] = l_dist_e
        if pd.notna(right_e):
            e[right_e] = r_dist_e
            (station_graph[right_e])[end] = r_dist_e

        if len(e) > 0:
            station_graph[end] = e

    return dijkstra(station_graph, start, end)

# 같은 호선의 두 역을 입력하여, 두 역 사이에 존재하는 역을 출력
def expand_path(start, end, merged):
    lines_s = set(merged.loc[merged["역명_cleaned"] == start, "호선"])
    lines_e = set(merged.loc[merged["역명_cleaned"] == end, "호선"])
    common_line = list(lines_s.intersection(lines_e))

    if not common_line:
        return None
    
    line = common_line[0]

    df_line = merged[merged["호선"] == line].drop_duplicates("역명_cleaned")

    order_s = df_line.loc[df_line["역명_cleaned"] == start, "stinConsOrdr"].iloc[0]
    order_e = df_line.loc[df_line["역명_cleaned"] == end, "stinConsOrdr"].iloc[0]

    if order_s <= order_e:
        seg = df_line[(df_line["stinConsOrdr"] >= order_s) &
                      (df_line["stinConsOrdr"] <= order_e)]
        
    else:
        seg = df_line[(df_line["stinConsOrdr"] >= order_e) &
                      (df_line["stinConsOrdr"] <= order_s)]
        
    seg = seg.sort_values("stinConsOrdr")["역명_cleaned"].tolist()
    return seg

# 한 역에서 다른 역으로 이동할 때, 경로상의 모든 역을 복원
def restore_path(adj_graph, start, end, merged):
    full = []

    result = find_path(adj_graph, start, end, merged)
    if result is None:
        return []

    path, _ = result
    for idx in range(len(path) - 1):
        subpath_s = path[idx]
        subpath_e = path[idx + 1]
        subpath = expand_path(subpath_s, subpath_e, merged)
        full.append(subpath)

    return full

# 공통 호선 추출
def find_common_line(stations, merged):
    common = None
    for st in stations:
        lines = set(merged.loc[merged["역명_cleaned"] == st, "호선"].unique())

        if common is None:
            common = lines
        else:
            common = common.intersection(lines)

    return list(common)[0]

# 역의 순서 추출
def get_order(station, line, merged):
    df = merged[(merged["역명_cleaned"] == station) & (merged["호선"] == line)]
    return df["stinConsOrdr"].iloc[0]

# 혼잡도 범위 추출
def calc_congestion_range(stations, merged):
    line = find_common_line(stations, merged)

    df_time = merged[(merged["시간"] >= 8.0) & (merged["시간"] < 9.0)]

    df_sel = df_time[(df_time["역명_cleaned"].isin(stations)) &
                     (df_time["호선"] == line)]
    
    order_s = get_order(stations[0], line, merged)
    order_e = get_order(stations[-1], line, merged)

    if order_s < order_e:
        if line == 2:
            direct = "외선"
        else:
            direct = "하선"

    else:
        if line == 2:
            direct = "내선"
        else:
            direct = "상선"

    df_direct = df_sel[df_sel["상하구분"] == direct]

    congestion_sum = df_direct.groupby("역명_cleaned")["혼잡도"].sum()
    return congestion_sum.max() - congestion_sum.min()

# 분류된 군집에 존재하는 모든 역을 반환
def get_stations_by_place(idx, merged):
    return merged[merged["place"] == idx]["역명_cleaned"].unique().tolist()

# 혼잡도의 변동성을 반환
def get_congestion_variance(adj_graph, start, end, merged):
    paths = restore_path(adj_graph, start, end, merged)
    if not paths:
        return 0
    
    ranges = [calc_congestion_range(p, merged) for p in paths]
    if not ranges:
        return 0
    return (max(ranges) - min(ranges))

# 군집 간 혼잡도 변동성의 리스트를 반환
def cong_list(adj_graph, idx1, idx2, merged):
    stations_1 = get_stations_by_place(idx1, merged)
    stations_2 = get_stations_by_place(idx2, merged)

    result = []
    for res in stations_1:
        for bus in stations_2:
            con = get_congestion_variance(adj_graph, res, bus, merged)
            result.append(con)

    return result

# 군집 간 혼잡도 변동성의 평균을 반환
def calc_congestion_mean(adj_graph, idx1, idx2, merged):
    con_list = cong_list(adj_graph, idx1, idx2, merged)
    n = len(con_list)
    for i in con_list:
        if i == 0:
            n -= 1

    if n == 0:
        return 0
    return sum(con_list) / n

line_colors = {
    1: "#0052A4",
    2: "#009D3E",
    3: "#EF7C1C",
    4: "#00A5DE",
    5: "#996CAC",
    6: "#CD7C2F",
    7: "#747F00",
    8: "#E6186C",
    12: "#009D3E",
    22: "#009D3E"
}

# 환승역이 저장된 df
transfer_dist = pd.read_csv("project/data/transfer_revised.csv", encoding = "utf-8")

remove_stations = ["까치산", "신설동"]
transfer_dist = transfer_dist[~transfer_dist["역명_cleaned"].isin(remove_stations)]

# 지하철역의 위치가 저장된 df
station_merged = pd.read_csv("project/data/staion_merged.csv", encoding = "utf-8")
station_merged["역명_cleaned"] = station_merged["지하철역"].apply(clean_name)

transfer_loc = transfer_dist.merge(station_merged, on = "역명_cleaned", how = "left")

line_groups = transfer_dist.groupby("lnCd")

# 이건뭐였더라
place_clustered = pd.read_csv("project/data/place_clustered.csv", encoding = "utf-8")
congest_clustered = pd.read_csv("project/data/congest_clustered.csv", encoding = "utf-8")

place_clustered["역명_cleaned"] = place_clustered["지하철역"].apply(clean_name)
pc = place_clustered.drop_duplicates(subset="역명_cleaned", keep="first")

merged = congest_clustered.merge(pc[["역명_cleaned", "place"]], on = "역명_cleaned", how = "left")

remove_stations = ["까치산", "신정네거리", "양천구청", "도림천",
                   "신설동", "용두", "신답", "용답"]

merged = merged[~((merged["역명_cleaned"].isin(remove_stations)) &
                  (merged["호선"] == 2))]

# 분류된 역들의 중심 위치가 저장된 df
centers = pd.read_csv("project/data/centers.csv", encoding = "utf-8")

# 연결된 환승역을 모두 저장
edges = []

for line, group in line_groups:
    group_sorted = group.sort_values("stinConsOrdr")

    stations = group_sorted["역명_cleaned"].tolist()
    orders = group_sorted["stinConsOrdr"].tolist()

    for i in range(len(stations) - 1):
        a = stations[i]
        b = stations[i + 1]
        dist = abs(orders[i] - orders[i + 1])

        edges.append((a, b, dist))

from collections import defaultdict
adj = defaultdict(list)

for a, b, d in edges:
    adj[a].append((b, d))
    adj[b].append((a, d))

adj = dict(adj)

adj_graph = {
    node: {neighbor: weight for neighbor, weight in adj_list}
    for node, adj_list in adj.items()
}

# 그래프에 표시하기 위해, 가장 가까운 연결된 환승역만을 저장
edges_line = []

for line, group in line_groups:
    group_sorted = group.sort_values("stinConsOrdr")
    stations = group_sorted["역명_cleaned"].tolist()

    for i in range(len(stations) - 1):
        a = stations[i]
        b = stations[i + 1]
        edges_line.append((a, b, line))

adj_line = defaultdict(list)

for a, b, d in edges_line:
    adj_line[a].append((b, d))
    adj_line[b].append((a, d))

adj_line = dict(adj_line)

adj_graph_line = {
    node: {neighbor: weight for neighbor, weight in adj_list}
    for node, adj_list in adj_line.items()
}

# 환승역들이 연결된 그래프 시각화
pos = {row["역명_cleaned"]: (row["lng"], row["lat"]) for _, row in transfer_loc.iterrows()}
min_lon = transfer_loc["lng"].min()
max_lon = transfer_loc["lng"].max()
min_lat = transfer_loc["lat"].min()
max_lat = transfer_loc["lat"].max()

lon_padding = (max_lon - min_lon) * 0.2
lat_padding = (max_lat - min_lat) * 0.2

edge_traces = []

for line in line_colors.keys():
    show_legend = False if line in [12, 22] else True
    line_edges = [(a, b) for a, b, ln in edges_line if ln == line]
    
    if len(line_edges) == 0:
        continue

    lons = []
    lats = []

    for a, b in line_edges:
        if a in pos and b in pos:
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            lons += [x0, x1, None]
            lats += [y0, y1, None]

    edge_traces.append(
        go.Scattergeo(
            lon=lons,
            lat=lats,
            mode="lines",
            line=dict(width=3, color=line_colors[line]),
            name=f"{line}호선",
            hoverinfo="none",
            showlegend=show_legend
        )
    )

node_trace = go.Scattergeo(
    lon=[pos[n][0] for n in pos.keys()],
    lat=[pos[n][1] for n in pos.keys()],
    mode="markers",
    text=list(pos.keys()),
    textposition="top center",
    marker=dict(size=6, color="black"),
    hoverinfo="text",
    name="역"
)

fig = go.Figure(edge_traces + [node_trace])

fig.update_layout(
    height=900,
    showlegend=True,
    legend=dict(font=dict(size=12)),
    geo=dict(
        projection_type="mercator",

        center=dict(lat=transfer_loc["lat"].mean(),
                    lon=transfer_loc["lng"].mean()),

        lonaxis=dict(range=[min_lon - lon_padding, max_lon + lon_padding]),
        lataxis=dict(range=[min_lat - lat_padding, max_lat + lat_padding]),

        showland=True,
        landcolor="rgb(240,240,240)"
    ),
    title="서울 지하철 네트워크"
)

# 역할 기반 분류 시각화
m1 = folium.Map(location=[37.5665, 126.9780], zoom_start=11)
filtered = station_merged[station_merged["role"] != "other"]

color_map = {
    "business": "red",
    "residential": "blue",
    "mixed": "green"
}

for _, row in filtered.iterrows():
    folium.CircleMarker(
        location = [row["lat"], row["lng"]],
        radius = 3,
        color = color_map.get(row["role"], "gray"),
        fill = True,
        fill_opacity = 0.7,
        popup = row["지하철역"]
    ).add_to(m1)

# 지리적 거리 기준 분류 시각화
m2 = folium.Map(location=[37.5665, 126.9780], zoom_start=10)

color_map = {
    "b": "red",
    "r": "blue",
    "m": "green"
}

for _, row in centers.iterrows():
    folium.Circle(
        location=[row["lat"], row["lng"]],
        radius=row["max"] / 2,
        color=color_map.get(row["type"], "gray"),
        fill=True,
        fill_opacity=0.7
    ).add_to(m2)

# 군집 간 연결 정도 시각화
df = pd.read_csv("project/data/result.csv", encoding = "utf-8")
results = {
    (row["from_cluster"], row["to_cluster"]): row["connection"]
    for _, row in df.iterrows()
}
TOP_N = 4
BOTTOM_N = 25
EXCLUDE = {8}
BLOCK = {1, 9}

m3 = folium.Map(location=[37.5665, 126.9780], zoom_start=11)

items = [(k, v) for k, v in results.items() if k[0] not in EXCLUDE and k[1] not in EXCLUDE]
items_sorted = sorted(items, key=lambda x: x[1])

bottom_items = items_sorted[:BOTTOM_N]      
top_items = items_sorted[-TOP_N:]           

vals = np.array([v for _, v in top_items + bottom_items], dtype=float)
vmin, vmax = vals.min(), vals.max()

def norm(x):
    return (x - vmin) / (vmax - vmin + 1e-9)

def get_color(t):
    R = int(255 * (1 - t))
    B = int(255 * t)
    return f"rgba({R},0,{B},0.7)"

coords = {row["place"]: (row["lat"], row["lng"]) for _, row in centers.iterrows()}

color_map = {"b": "red", "r": "blue", "m": "green"}

for _, row in centers.iterrows():
    if row["place"] in EXCLUDE:
        continue

    folium.Circle(
        location=[row["lat"], row["lng"]],
        radius=row["max"] / 2,
        color=color_map.get(row["type"], "gray"),
        fill=True,
        fill_opacity=0.4,
        weight=3
    ).add_to(m3)

for (i, j), val in bottom_items:
    t = norm(val)
    if i in BLOCK or j in BLOCK:
        continue
    color = get_color(t)
    lat1, lon1 = coords[i]
    lat2, lon2 = coords[j]

    folium.PolyLine(
        locations=[(lat1, lon1), (lat2, lon2)],
        color=color,
        weight=4,
        opacity=1
    ).add_to(m3)

for (i, j), val in top_items:
    t = norm(val)
    if i in BLOCK or j in BLOCK:
        continue
    color = get_color(t)
    lat1, lon1 = coords[i]
    lat2, lon2 = coords[j]

    folium.PolyLine(
        locations=[(lat1, lon1), (lat2, lon2)],
        color=color,
        weight=4,
        opacity=1
    ).add_to(m3)

# 스트림릿
st.set_page_config(page_title="서울 지하철 분석", layout="wide")

st.sidebar.title("메뉴")
page = st.sidebar.radio(
    "페이지를 선택하세요",
    ["홈", "탐색적 데이터 분석 (EDA)", "프로젝트 진행 과정", "결론 및 참고사항"]
)

def show_home():
    st.title("서울 지하철 혼잡도와 승하차 행동 기반 군집화 및 네트워크 분석")

    st.markdown("""
                ### 프로젝트 개요
                본 프로젝트는 시간대별 혼잡도 패턴, 승하차수 기반 지하철역 역할 분류, 지하철 네트워크 구조를 종합적으로 분석하여
                서울 도시 이동 흐름을 이해하는 것을 목표로 한다.

                ### 분석 목표
                - 출근·퇴근시간대 혼잡도 패턴 분석
                - 8-means clustering을 통한 역 이용행태 분류
                - 역 분포의 공간적 패턴 파악
                - 공간적 패턴을 기반으로 한 이동 구조 이해

                ### 페이지 구성
                - **탐색적 데이터 분석 (EDA)**:
                시간대, 호선별 혼잡도 시각화, 승하차수 기반 clustering, Google Places 기반 역할 검증

                - **프로젝트 진행 과정**:
                지리적 위치 기준 재분류, 역 간 최단경로 계산, 군집 간 연결 정도 시각화

                - **결론 및 참고사항**:
                결론 및 한계점, 참고사항
                ---
                """)

def show_eda():
    st.title("탐색적 데이터 분석 (EDA)")

    st.subheader("1. 시간대별 평균 혼잡도 분석")
    st.write("시간대별 혼잡도 시각화")
    st.image("project/data/peak.png", caption = "시간대별 혼잡도", use_container_width = True)

    st.write("호선별 혼잡도 시각화")

    col1, col2 = st.columns(2)

    with col1:
        st.image("project/data/go_cong.png", use_container_width = True)
    with col2:
        st.image("project/data/come_cong.png", use_container_width = True)
    st.caption("호선별 혼잡도")

    st.write("""
             총 혼잡도는 8시 (출근시간대)와 18시 (퇴근시간대)에 피크가 나타난다.
             호선의 경우 2호선의 혼잡도가 가장 높고, 6호선의 혼잡도가 가장 낮다.
             """)

    st.subheader("2. 승하차수 기반 역 클러스터링")
    st.write("혼잡도 피크가 나타났던 8시, 18시에서의 승하차인원 수를 기준으로 8-means clustering을 진행하였다.")
    st.write("clustering 결과")
    
    data = {
    "08시-09시 승차인원": [1.186411e+06, 2.159950e+07, 8.948420e+06, 5.947758e+06, 4.314955e+06, 1.434197e+07, 4.271044e+06, 3.363399e+06],
    "08시-09시 하차인원": [1.106536e+06, 1.933667e+07, 4.679640e+06, 5.018022e+07, 3.136264e+06, 7.946114e+06, 2.568294e+07, 1.323437e+07],
    "18시-19시 승차인원": [1.062314e+06, 2.442245e+07, 4.861536e+06, 4.306625e+07, 3.028475e+06, 9.173851e+06, 2.328459e+07, 1.180764e+07],
    "18시-19시 하차인원": [1.094503e+06, 2.931356e+07, 8.075084e+06, 1.295528e+07, 3.880769e+06, 1.483412e+07, 9.017484e+06, 5.342251e+06],
    }
    df_cluster = pd.DataFrame(data)
    df_cluster = df_cluster.astype(int)
    st.dataframe(df_cluster.style.format("{:,.0f}"))

    st.write("""
             출근시간대에 승차인원이 많고, 퇴근시간대에 하차인원이 많을수록 거주지구의 성격을 띄며, 
             출근시간대에 하차인원이 많고, 퇴근시간대에 승차인원이 많을수록 업무지구의 성격을 띌 것이다.
             이를 기준으로 3, 6번을 업무지구, 5번을 거주지구, 1번을 혼합지구로 분류하였다.
             """)
    
    st.write("분류 결과")
    col1, col2 = st.columns(2)

    with col1:
        st.image("project/data/clustered_8.png", use_container_width = True)
    with col2:
        st.image("project/data/clustered_18.png", use_container_width = True)
    st.caption("승하차인원에 따른 분류 결과")
    
    st.write("""
             분류가 올바르게 수행되었는지 확인하기 위해, Google Map API를 활용해 웹크롤링으로써
             각 지하철역 주변에 분포하는 업무 시설 및 거주 편의시설의 개수를 수집하였다.
             지하철역은 거주지구의 성격을 강하게 띄는 당산역과 연신내역,
             업무지구의 성격을 강하게 띄는 시청역과 서울역에 대해 조사하였다.
             """)
    
    st.write("수집 결과")

    df_zone = pd.DataFrame({
        "역": ["당산역", "연신내역", "시청역", "서울역"],
        "업무 시설 개수": [57, 59, 89, 47],
        "거주 편의시설 개수": [36, 40, 37, 23]
    })
    st.dataframe(df_zone)
    st.bar_chart(df_zone.set_index("역"))
    st.write("거주지구에서 업무지구에 비해 상대적으로 거주 편의시설 개수의 비율이 높은 것을 확인할 수 있다.")

    st.write("분류 결과")
    st_folium(m1, width=900, height=600)
    st.caption("업무지구(빨간색), 거주지구(파란색), 혼합지구(초록색) 분류 결과")
    st.write("주요 업무지구로 알려진 종로, 강남 등에 빨간색 점이 밀집되어 분포하는 것을 확인할 수 있다.")

def show_process():
    st.title("프로젝트 진행 과정")
    st.subheader("1. 공간 위치 기준 clustering")
    st.write("""
             EDA 과정 때와 마찬가지로, k-means clustering을 이용하여 지리적 공간을 기준으로, 업무지구/거주지구/혼합지구를 재분류하였다.
             분류된 군집들의 각 중심점을 계산한 뒤, 중심으로부터의 최대 거리를 기준으로 지리적 범위를 설정하여 지도에 시각화를 진행한다.
             """)
    
    st.write("이상치 제거")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("project/data/clustered_b.png", use_container_width = True)
    with col2:
        st.image("project/data/clustered_r.png", use_container_width = True)
    with col3:
        st.image("project/data/clustered_m.png", use_container_width = True)
    st.caption("중심으로부터의 거리 분포")

    st.write("""
             박스플랏에서 확인된 이상치들을 제거한 후, 중심 기준 최대 거리를 계산하였다.
             """)
    
    st.write("분류 결과")
    st_folium(m2, width=900, height=600)
    st.write("EDA 탭에서 수행된 분류 결과와 비슷한 패턴을 나타낸다.")

    st.subheader("2. 역 간 이동 경로 탐색")
    st.write("""
             역 간 이동 패턴을 파악하기 위해 이동 경로를 복원한다.
             이를 수행하기 위해 환승역을 기반으로 환승역 간의 거리가 저장된 가중치 그래프를 정의한다.
             역 간의 연결 관계와 순서는 코레일에서 API 키를 발급받아 크롤링하였으며, 
             이를 기반으로 역 간 거리 정보를 정의하였다.
             """)
    
    st.write("환승역 기반 그래프 생성 결과")
    st.plotly_chart(fig)
    st.caption("환승역 기반 그래프")

    st.write("""
             그래프 생성 후, 환승역이 아닌 역에 대하여 노선 상 양방향에 위치한 두 개의 환승역과 각 환승역까지의 거리를 모두 저장한다.
             이를 기반으로 출발역과 도착역이 입력되면, 
             각각에서 가장 가까운 환승역을 그래프에 추가하여 최종 그래프를 구성한다.
             이후 해당 그래프에 다익스트라 알고리즘을 적용하여 출발역에서 도착역까지의 최단 경로를 탐색한다.

             아래에 출발역과 도착역을 입력하면 최단 경로를 확인할 수 있다.
             """)
    
    start = st.text_input("출발역을 입력하세요.")
    end = st.text_input("도착역을 입력하세요.")
    st.caption("1~8호선에 대해서만 입력 가능, 1호선 일부 역은 누락되었을 수 있음.")

    if st.button("최단 경로 탐색"):
        if not start or not end:
            st.warning("출발역과 도착역을 모두 입력하세요.")
        else:
            start_cleaned = clean_name(start)
            end_cleaned = clean_name(end)
            stations = list(set(merged["역명_cleaned"].tolist()))
            if start_cleaned not in stations or end_cleaned not in stations:
                st.error("존재하지 않는 역 이름입니다.")
            else:
                path, dist = find_path(adj_graph, start_cleaned, end_cleaned, merged)

                st.success("최단경로 탐색 완료")
                st.write(" → ".join(path))
                st.write(f"총 이동 거리: {dist}")

    st.write("""
             본 분석은 1~8호선을 대상으로 수행되었으며, 9호선·수인분당선 등 기타 노선은 데이터에 포함되지 않았다.
             또한 경로 탐색 과정에서 경유하는 역의 개수를 거리 지표를 사용하였기에 
             실제 물리적 거리, 환승 횟수와 같은 요소를 직접적으로 반영하지 못하는 한계가 존재한다.
             그럼에도 불구하고, 주요 이동 경로를 합리적으로 설명하며 전반적으로 유의미한 성능을 보이는 것을 확인할 수 있었다.
             """)
    
    st.subheader("3. 공간별 이동 패턴 파악")
    st.write("""
             위 경로 탐색 알고리즘을 기반으로 도출된 경로는 환승역만을 포함한다.
             그러나 역 간 이동 패턴을 파악하기 위해서 역을 이동할 때 전체 경로가 복원되어야 한다.
             이를 수행하기 위해 2. 역 간 이동 경로 탐색에서 크롤링했던 역 순서 데이터를 다시 활용하였다.
             같은 호선 상에서 이동하는 경우, 출발역과 도착역의 역 순서 정보를 추출한 뒤, 
             두 역의 순서 값 사이에 위치한 모든 역을 경로에 포함시키는 과정으로 환승역 사이의 세부 이동을 복원하였다.
             """)
    st.write("")
    st.write("""
             다음으로 출근시간대의 역 간 연결 정도를 정량적으로 계산하였다.
             본 프로젝트에서 역 간의 연결 정도는, 출근시간대의 혼잡도의 변화 양상을 기준으로 정의하였다.

             역 간의 연결 정도가 큰 경우, 승객들이 중간 역에서 하차하지 않고 이동하는 경향이 많으므로 
             연속된 경로에서 혼잡도의 분산이 상대적으로 작게 나타날 것으로 가정하였다.
             반대로 역 간의 연결 정도가 작은 경우, 
             승객들이 중간 역에서 하차하는 승객의 비율이 높아 역 간 혼잡도의 변동성이 크게 나타날 것으로 예상하였다.

             1. 에서 수행했던 지리적 위치 기준 분류된 결과를 활용하여,
             공간 군집 간의 연결 정도를 계산하였다.

             구체적으로, 하나의 군집에서 다른 군집으로 이동하는 경우를 정의하고, 
             해당 군집에 포함된 모든 역 쌍(출발 군집의 역 → 도착 군집의 역)에 대해 
             출근시간대 혼잡도의 변동성을 계산하였다.
             이때 각 역 쌍에서 산출된 혼잡도 변동성을 평균하여, 
             두 군집 간의 연결 정도를 하나의 값으로 정의하였다.
             
             복원된 전체 경로와, 지하철 혼잡도 데이터를 결합하여 
             군집 간 연결 정도를 계산하였으며,
             그 시각화 결과는 아래와 같다.
             """)
    st_folium(m3, width=900, height=600)
    st.caption("군집 간 연결 정도 시각화. 연결 정도가 높을수록 빨간색으로, 낮을수록 파란색으로 표현하였다.")
    st.write("")
    st.write("""
             가까운 군집 간 연결 정도가 높고, 먼 군집 간 연결 정도가 낮은 것으로 보아 시각화가 잘 진행됐음을 확인할 수 있다.
             """)
    
def show_conclusion():
    st.title("결론 및 참고사항")
    st.markdown(
        """
        ### 결론

        본 프로젝트는 서울 지하철 데이터를 기반으로 시간대별 혼잡도, 승하차 행태,
        그리고 지하철 네트워크 구조를 종합적으로 분석하여 도시 이동 패턴을 해석하고자 하였다.

        환승역 중심의 가중치 그래프와 경로 복원 알고리즘을 통해
        역 간 이동을 단순한 최단 거리 문제가 아닌,
        실제 이동 흐름을 반영한 네트워크 문제로 재구성하였다.

        특히 혼잡도 변동성을 활용하여 공간 군집 간 연결 정도를 정의함으로써, 
        지리적으로 인접한 지역 간 이동은 강하게 연결되고,
        원거리 지역 간 이동은 상대적으로 약하게 연결되는
        도시 이동 구조의 특성을 정량적으로 확인할 수 있었다.

        이러한 결과는 지하철 혼잡도 데이터가 단순한 혼잡 지표를 넘어, 
        도시 공간 간 상호작용을 설명하는 지표로 활용될 수 있음을 시사한다.

        ### 한계점

        - 본 분석은 1~8호선 데이터를 대상으로 수행되었으며 9호선, 수인분당선 등 기타 노선은 데이터에 포함되지 않았다.
        9호선의 경우 혼잡도 데이터가 존재하였으나, 혼잡도 산정 기준이 1~8호선과 상이하여 
        동일한 기준 하에서의 정량적 비교가 어렵다고 판단하였다.
        이에 따라 혼잡도 지표의 일관성을 유지하기 위해 1~8호선 데이터만을 활용하여 분석을 수행하였다.

        - 1호선의 경우, 노선 특성상 서울교통공사와 한국교통공사가 공동 운영하고 있으며, 
        본 프로젝트의 활용 데이터에는 서울교통공사 소속 구간의 데이터만이 포함되었다.
        따라서 1호선 전체 노선을 포괄하지 않으며, 1호선 전반의 이동 패턴을 완전히 반영하지는 못한다.

        - 역 간 이동 거리는 실제 물리적 거리나 소요 시간을 사용하지 않고, 
        경유하는 역의 개수를 기반으로 정의하였다.
        이로 인해 실제 이동 시간이나 환승 횟수는 직접적으로 고려되지 않았다.

        그럼에도 불구하고, 본 분석은 주요 이동 경로와 공간 군집 간의 관계를
        일관된 기준 하에서 설명하며, 전반적으로 유의미한 분석 결과를 도출하였다.

        ### 참고사항

        본 프로젝트의 데이터 분석 과정에서, 지하철역 주변의 시설 분포 확인 및 누락된 위도, 경도를 채워넣기 위해 Google Maps API를 활용하였다.
        다만 보안을 위해 Google Maps API는 최종 코드에서 제외하였으며, 필요시 project2.ipynb의 파일이 7번 셀에 API를 넣는다면 분석 과정을 재현할 수 있다.

        Google Maps API가 활용된 코드는 다음과 같다.
        - 지하철역 주변의 시설 분포 확인 (project2.ipynb 파일의 9번 셀)

        - 누락된 위도 및 경도 삽입 (project2.ipynb 파일의 26번 셀)

        Google Maps API를 이용해 보완된 위도·경도 정보는 
        분석 과정의 재현성과 계산 효율을 고려하여 
        별도의 csv 파일로 저장한 뒤 이후 분석 단계에서 재활용하였다.
        이에 따라 누락된 위도·경도를 보완하는 과정은
        API 키 없이도 해당 csv 파일로 재현이 가능하다.

        보완된 해당 csv 파일은 project/data 폴더 내 staion_merged.csv 파일로 저장하였다.
        """)

if page == "홈":
    show_home()

elif page == "탐색적 데이터 분석 (EDA)":
    show_eda()

elif page == "프로젝트 진행 과정":
    show_process()

elif page == "결론 및 참고사항":
    show_conclusion()
