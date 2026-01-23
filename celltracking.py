import numpy as np
import pandas as pd
import trackpy as tp
from cellpose import models
from skimage.measure import regionprops
from tifffile import imread


FRAME_INTERVAL_MIN = 5   
MAX_DISPLACEMENT = 25            
TRACK_MEMORY = 3                   
EDGE_MARGIN = 20                   
MIN_TRACK_LENGTH = 4               
DIV_AREA_TOL = 0.25                
STASIS_GROWTH_THRESH = 0.02        


model = models.CellposeModel(
    gpu=True,
    model_type='omnipose'
)

def omnipose_segment(stack):

    print("Segmenting with Omnipose...")

    masks, flows, styles = model.eval(
        stack,
        diameter=None,
        channels=[0, 0]
    )

    return masks



def masks_to_df(mask_stack):

    rows = []

    for t, mask in enumerate(mask_stack):

        props = regionprops(mask)

        for p in props:

            y, x = p.centroid

            rows.append({
                'frame': t,
                'x': x,
                'y': y,
                'area': p.area,
                'eccentricity': p.eccentricity
            })

    return pd.DataFrame(rows)



def track_cells(df):

    print("Linking trajectories...")

    linked = tp.link_df(
        df,
        search_range=MAX_DISPLACEMENT,
        memory=TRACK_MEMORY
    )

    # Remove short-lived noise objects
    linked = tp.filter_stubs(linked, threshold=MIN_TRACK_LENGTH)

    return linked



def is_near_border(x, y, shape):

    h, w = shape
    return (
        x < EDGE_MARGIN or
        y < EDGE_MARGIN or
        x > w - EDGE_MARGIN or
        y > h - EDGE_MARGIN
    )


def classify_events(linked, image_shape):

    births = []
    deaths = []
    divisions = []

    max_frame = linked.frame.max()

    for pid, track in linked.groupby('particle'):

        track = track.sort_values('frame')

        frames = track.frame.values
        areas = track.area.values

        start = frames[0]
        end = frames[-1]

        x_end = track.iloc[-1].x
        y_end = track.iloc[-1].y


        if start > TRACK_MEMORY:

            births.append({
                'particle': pid,
                'frame': start
            })


        if end < max_frame:

            if not is_near_border(x_end, y_end, image_shape):

                deaths.append({
                    'particle': pid,
                    'frame': end
                })


        if len(areas) >= 3:

            area_change = np.diff(areas) / areas[:-1]

            if np.min(area_change) < -DIV_AREA_TOL:
                divisions.append({
                    'particle': pid,
                    'frame': frames[np.argmin(area_change)+1]
                })

    return births, deaths, divisions


def classify_stasis(linked):

    stasis_cells = []
    active_cells = []

    for pid, track in linked.groupby('particle'):

        track = track.sort_values('frame')

        t_hours = (track.frame.values * FRAME_INTERVAL_MIN) / 60
        areas = track.area.values

        if len(areas) < 3:
            continue

        growth_rate = np.polyfit(t_hours, areas, 1)[0] / np.mean(areas)

        if abs(growth_rate) < STASIS_GROWTH_THRESH:
            stasis_cells.append(pid)
        else:
            active_cells.append(pid)

    return stasis_cells, active_cells


def run_full_pipeline(image_stack):

    masks = omnipose_segment(image_stack)

    df = masks_to_df(masks)

    linked = track_cells(df)

    births, deaths, divisions = classify_events(
        linked,
        image_shape=image_stack[0].shape
    )

    stasis_cells, active_cells = classify_stasis(linked)

    print("\n===== FINAL RESULTS =====")
    print("Total tracked cells:", linked.particle.nunique())
    print("Birth events:", len(births))
    print("Death events:", len(deaths))
    print("Division events:", len(divisions))
    print("Stasis cells:", len(stasis_cells))
    print("Active cells:", len(active_cells))

    return linked, births, deaths, divisions, stasis_cells
