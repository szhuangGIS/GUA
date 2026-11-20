#!/usr/bin/env python
# coding: utf-8

# 1. Read the panel inputs

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import shape, box
from shapely.ops import unary_union
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Polygon, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'Data'
panel_sheets = pd.read_excel(
    DATA_DIR / '05_Figure3_Source_Data.xlsx',
    sheet_name=['Fig3a', 'Fig3b', 'Fig3c', 'Fig3d', 'Fig3e', 'Fig3f', 'Fig3g'],
)
boundary_path = DATA_DIR / 'Map_Boundary' / 'WB_GAD_ADM0_Income.shp'
country_background = pd.read_excel(DATA_DIR / '05_Figure3_Country_Background.xlsx', sheet_name='Data')

TARGETS = ['Frequency', 'MBEP']
GDP_GROUPS = ['Slower', 'Middle', 'Faster']
GROUP_COLORS = {'Slower': '#3C5488', 'Middle': '#00A087', 'Faster': '#E64B35', 'Unclassified': '#B8B8B8'}
MAP_LIMITS = {'Frequency': 0.3, 'MBEP': 0.6}
VIOLIN_BANDWIDTH = 'scott'
palette_rgb = [[36, 0, 216], [24, 28, 247], [40, 87, 255], [61, 135, 255], [86, 176, 255], [117, 211, 255], [153, 234, 255], [188, 249, 255], [234, 255, 255], [255, 255, 234], [255, 241, 188], [255, 214, 153], [255, 172, 117], [255, 120, 86], [255, 61, 61], [247, 39, 53], [216, 21, 47], [165, 0, 33]]
ue_cmap = LinearSegmentedColormap.from_list('BlueDarkRed18', np.array(palette_rgb) / 255)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'legend.fontsize': 6,
    'figure.dpi': 190, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
})


# 2. Organize the country maps, rankings and relationship

# Country sample counts determine bubble sizes, not regression weights.
country_sample_sizes = {'ARG': 80,
 'AUS': 36,
 'BGD': 8,
 'BLZ': 10,
 'BOL': 11,
 'BRA': 466,
 'CAN': 62,
 'CHN': 246,
 'COL': 63,
 'CUB': 36,
 'EGY': 11,
 'ESP': 15,
 'FIN': 7,
 'FRA': 33,
 'GRC': 2,
 'HND': 2,
 'HTI': 3,
 'IND': 47,
 'ISL': 1,
 'ITA': 31,
 'KAZ': 33,
 'KHM': 25,
 'KOR': 1,
 'LKA': 10,
 'MAR': 1,
 'MDG': 63,
 'MEX': 55,
 'MMR': 4,
 'MOZ': 13,
 'PAK': 13,
 'PER': 2,
 'PRT': 10,
 'PRY': 9,
 'ROU': 5,
 'RUS': 113,
 'RWA': 2,
 'SEN': 3,
 'SWE': 1,
 'THA': 47,
 'TUR': 13,
 'UKR': 32,
 'URY': 16,
 'USA': 526,
 'VEN': 36,
 'VNM': 4,
 'ZAF': 3,
 'ZMB': 2,
 'ZWE': 4}

map_data = {}
map_group_lookup = {}
for target, sheet_name in [('Frequency', 'Fig3a'), ('MBEP', 'Fig3b')]:
    frame = panel_sheets[sheet_name].copy()
    frame['N_lakes'] = frame['ISO_A3'].map(country_sample_sizes)
    map_group_lookup[target] = country_background.set_index('ISO_A3')['Country_GDP_group'].to_dict()
    map_data[target] = frame

ranking_frames = []
for target, sheet_name in [('Frequency', 'Fig3e'), ('MBEP', 'Fig3f')]:
    frame = panel_sheets[sheet_name].sort_values('Mean_UE_pp_per_year', ascending=False).copy()
    frame['Target'] = target
    frame['Rank'] = np.arange(1, len(frame) + 1)
    ranking_frames.append(frame)
rankings = pd.concat(ranking_frames, ignore_index=True)

country = panel_sheets['Fig3g'].rename(columns={
    'Frequency_UE_pp_per_year': 'Frequency', 'MBEP_UE_pp_per_year': 'MBEP'
}).copy()
country['N_lakes'] = country['ISO_A3'].map(country_sample_sizes)
slope, intercept = np.polyfit(country['Frequency'], country['MBEP'], 1)
correlation, p_value = pearsonr(country['Frequency'], country['MBEP'])
relationship = pd.DataFrame([{
    'N_countries': len(country), 'N_lakes': country['N_lakes'].sum(),
    'Pearson_r': correlation, 'Pearson_p_two_sided': p_value,
    'OLS_slope': slope, 'OLS_intercept': intercept,
}])


# 3. Calculate the lake-group distribution summaries

distribution_frames = []
for target, sheet_name in [('Frequency', 'Fig3c'), ('MBEP', 'Fig3d')]:
    frame = panel_sheets[sheet_name].copy()
    frame['Target'] = target
    distribution_frames.append(frame)
plot_data = pd.concat(distribution_frames, ignore_index=True)

records = []
for target in TARGETS:
    for group in GDP_GROUPS:
        selected = plot_data['Target'].eq(target) & plot_data['Lake_GDP_group'].eq(group)
        values = plot_data.loc[selected, 'UE_pp_per_year'].to_numpy()
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        within_whiskers = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
        positive_values = values[values > 0]
        negative_values = values[values < 0]
        records.append({
            'Target': target, 'Lake_GDP_group': group, 'n': len(values),
            'mean': values.mean(), 'median': median, 'p25': q1, 'p75': q3,
            'min': values.min(), 'max': values.max(),
            'whisker_low': within_whiskers.min(), 'whisker_high': within_whiskers.max(),
            'outlier_n': len(values) - len(within_whiskers),
            'positive_n': len(positive_values), 'negative_n': len(negative_values),
            'zero_n': np.count_nonzero(values == 0),
            'positive_mean': positive_values.mean() if len(positive_values) else np.nan,
            'negative_mean': negative_values.mean() if len(negative_values) else np.nan,
        })
stats = pd.DataFrame(records)


# 4. Prepare the local map geometry

source_crs = CRS.from_wkt(boundary_path.with_suffix('.prj').read_text())
projection = Transformer.from_crs(source_crs, '+proj=robin +datum=WGS84 +units=m', always_xy=True)
for frame in map_data.values():
    frame['Map_x_m'], frame['Map_y_m'] = projection.transform(
        frame['Longitude'].to_numpy(), frame['Latitude'].to_numpy()
    )

country_parts = {}
with shapefile.Reader(str(boundary_path)) as reader:
    for item in reader.iterShapeRecords():
        iso = item.record['ISO_A3']
        geometry = shape(item.shape.__geo_interface__)
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        country_parts.setdefault(iso, []).append(geometry)

clip_extent = box(-180, -60, 180, 85)
world_paths = []
for iso, geometries in country_parts.items():
    geometry = unary_union(geometries).intersection(clip_extent)
    geometry = geometry.simplify(0.035, preserve_topology=True)
    parts = [geometry] if geometry.geom_type == 'Polygon' else list(getattr(geometry, 'geoms', []))
    for polygon in parts:
        if polygon.geom_type != 'Polygon' or polygon.is_empty or polygon.area < 0.005:
            continue
        vertices, path_codes = [], []
        for ring in [polygon.exterior, *polygon.interiors]:
            coordinates = np.asarray(ring.coords)
            x, y = projection.transform(coordinates[:, 0], coordinates[:, 1])
            vertices.extend(np.column_stack([x, y]))
            path_codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(coordinates) - 2) + [MplPath.CLOSEPOLY])
        world_paths.append((iso, MplPath(vertices, path_codes)))

outline_longitudes = np.concatenate([
    np.linspace(-180, 180, 361), np.full(146, 180),
    np.linspace(180, -180, 361), np.full(146, -180),
])
outline_latitudes = np.concatenate([
    np.full(361, -60), np.linspace(-60, 85, 146),
    np.full(361, 85), np.linspace(85, -60, 146),
])
outline_x, outline_y = projection.transform(outline_longitudes, outline_latitudes)
outline = np.column_stack([outline_x, outline_y])
map_fills = {group: tuple(0.28 * np.array(to_rgb(color)) + 0.72) for group, color in GROUP_COLORS.items()}
map_fills['Unclassified'] = '#EEEEEE'
summary_handles = [
    Line2D([], [], marker='D', ls='', color='#222222', markerfacecolor='white', markeredgewidth=0.8, markersize=4, label='Mean'),
    Line2D([], [], color='#222222', lw=1.1, label='Median'),
]


# 5. Draw the country maps and group distributions

def label_panel(ax,letter,x=-.04,y=1.03):
    ax.text(x,y,letter,transform=ax.transAxes,fontweight='bold',fontsize=12,va='bottom')


def draw_map(ax,target,show_legend=True,cax=None):
    for iso,path in world_paths:
        group = map_group_lookup[target].get(iso,'Unclassified')
        ax.add_patch(PathPatch(path,facecolor=map_fills[group],edgecolor='white',lw=.2,zorder=1))
    ax.add_patch(Polygon(outline,closed=True,fill=False,edgecolor='#444444',lw=.6,zorder=2))
    norm = TwoSlopeNorm(vmin=-MAP_LIMITS[target],vcenter=0,vmax=MAP_LIMITS[target])
    for positive,z in [(False,3),(True,4)]:
        d = map_data[target].loc[map_data[target]['Mean_UE_pp_per_year'].ge(0).eq(positive)].sort_values('N_lakes',ascending=False)
        ax.scatter(d.Map_x_m,d.Map_y_m,c=d.Mean_UE_pp_per_year,s=3*np.sqrt(d.N_lakes),cmap=ue_cmap,norm=norm,
            alpha=.95,edgecolors='#606060',linewidths=.35,zorder=z)
    ax.set(xlim=(outline[:,0].min(),outline[:,0].max()),ylim=(outline[:,1].min(),outline[:,1].max()),aspect='equal')
    ax.axis('off')
    if show_legend:
        handles=[Patch(facecolor=map_fills[g],label=g) for g in [*GDP_GROUPS,'Unclassified']]
        legend=ax.legend(handles=handles,title='Country GDP growth',loc='lower left',
            bbox_to_anchor=(.07,.15),fontsize=5.5,title_fontsize=6,handlelength=1.5,labelspacing=.3)
        ax.add_artist(legend)
        size_handles=[ax.scatter([],[],s=3*np.sqrt(n),facecolor='none',edgecolors='#777777',lw=.4,label=str(n)) for n in [10,100,500]]
        ax.legend(handles=size_handles,title='Lakes',loc='lower right',bbox_to_anchor=(.78,.07),
            fontsize=5.5,title_fontsize=6,labelspacing=.6,handletextpad=.4)
    if cax is None:
        cb=ax.figure.colorbar(ScalarMappable(norm=norm,cmap=ue_cmap),ax=ax,orientation='horizontal',
            pad=.035,fraction=.045,shrink=.55,extend='both')
    else:
        cb=ax.figure.colorbar(ScalarMappable(norm=norm,cmap=ue_cmap),cax=cax,
            orientation='horizontal',extend='both')
    cb.set_ticks([-MAP_LIMITS[target],0,MAP_LIMITS[target]])
    cb.ax.tick_params(length=2,labelsize=6);cb.outline.set_linewidth(.5)
    cb.set_label(f'UE on {"Frequency" if target=="Frequency" else "MBEP"} (pp yr$^{{-1}}$)',labelpad=2,fontsize=6.5)


def draw_column(ax_dist,ax_prop,target,letter=None):
    target_stats = stats.loc[stats.Target.eq(target)].set_index('Lake_GDP_group').loc[GDP_GROUPS]
    full_values = plot_data.loc[plot_data.Target.eq(target),'UE_pp_per_year'].to_numpy()
    span = np.ptp(full_values)
    ax_dist.set_ylim(full_values.min()-.055*span,full_values.max()+.13*span)
    ax_dist.axhline(0,color='#888888',ls='--',lw=.7,zorder=0)
    for x,group in enumerate(GDP_GROUPS):
        v = plot_data.loc[plot_data.Target.eq(target)&plot_data.Lake_GDP_group.eq(group),'UE_pp_per_year'].to_numpy()
        c = GROUP_COLORS[group]
        violin = ax_dist.violinplot(v,positions=[x],widths=.74,points=250,
            showmeans=False,showmedians=False,showextrema=False,bw_method=VIOLIN_BANDWIDTH)
        for body in violin['bodies']:
            body.set_facecolor(c);body.set_edgecolor(c);body.set_alpha(.28);body.set_linewidth(.7)
        ax_dist.boxplot(v,positions=[x],widths=.14,whis=1.5,patch_artist=True,manage_ticks=False,
            boxprops={'facecolor':'white','edgecolor':c,'linewidth':.8},
            whiskerprops={'color':c,'linewidth':.8},capprops={'color':c,'linewidth':.8},
            medianprops={'color':'#222222','linewidth':1.1},
            flierprops={'marker':'.','markersize':2.2,'markerfacecolor':c,
                'markeredgecolor':'none','alpha':.45},zorder=3)
        ax_dist.scatter(x+.17,target_stats.loc[group,'mean'],marker='D',s=17,
            facecolor='white',edgecolor='#222222',linewidth=.8,zorder=5)
    ax_dist.set_xlim(-.55,2.55)
    ax_dist.set_xticks(range(3));ax_dist.tick_params(axis='x',bottom=False,labelbottom=False)
    ax_dist.spines['bottom'].set_visible(False)
    ax_dist.yaxis.set_major_locator(MaxNLocator(6))
    ax_dist.set_ylabel(f'UE on {"frequency" if target=="Frequency" else "MBEP"} (pp yr$^{{-1}}$)',labelpad=5)
    if target=="Frequency":
        ax_dist.set_title(target,pad=9)
    else:
        ax_dist.set_title("MBEP",pad=9)
    if letter is not None: label_panel(ax_dist,letter,x=-.23)

    draw_strength(ax_prop,target)


def draw_strength(ax,target):
    d = stats.loc[stats.Target.eq(target)].set_index('Lake_GDP_group').loc[GDP_GROUPS]
    pos = d.positive_mean.to_numpy()
    neg = d.negative_mean.to_numpy()
    limit = np.ceil(np.nanmax(np.abs(np.r_[pos,neg]))*1.35/.1)*.1
    ax.bar(range(3),pos,width=.55,color=[GROUP_COLORS[g] for g in GDP_GROUPS],
        alpha=.80,edgecolor='white',linewidth=.5,zorder=2)
    ax.bar(range(3),neg,width=.55,color=[GROUP_COLORS[g] for g in GDP_GROUPS],
        alpha=.45,edgecolor='white',linewidth=.5,zorder=2)
    ax.axhline(0,color='#555555',lw=.75,zorder=3)
    for x,(positive,negative) in enumerate(zip(pos,neg)):
        if np.isfinite(positive):
            ax.text(x,positive+.035*limit,f'{positive:+.3f}',ha='center',va='bottom',fontsize=5.8)
        if np.isfinite(negative):
            ax.text(x,negative-.035*limit,f'{negative:.3f}',ha='center',va='top',fontsize=5.8)
    ax.set_ylim(-limit,limit);ax.set_xlim(-.55,2.55)
    ax.yaxis.set_major_locator(MaxNLocator(5,symmetric=True))
    ax.set_xticks(range(3));ax.set_xticklabels(GDP_GROUPS)
    ax.set_xlabel('Lake GDP growth group',labelpad=6)
    ax.set_ylabel('Mean positive / negative UE\n(pp yr$^{-1}$)',labelpad=5,fontsize=6.5)
    ax.tick_params(axis='x',length=0,pad=5)


# 6. Draw the rankings and country relationship

short_names = {'United States of America': 'U.S.A.', 'Russian Federation': 'Russia', 'Venezuela (Bolivarian Republic of)': 'Venezuela', 'Bolivia (Plurinational State of)': 'Bolivia', 'Republic of Korea': 'South Korea'}

def draw_radial(ax,target,cax=None):
    d=rankings.loc[rankings.Target.eq(target)].sort_values('Rank')
    values=d.Mean_UE_pp_per_year.to_numpy(); n=len(d)
    theta=np.arange(n)*2*np.pi/n
    scale=.72/max(np.max(np.abs(values)),1e-12)
    limit = MAP_LIMITS[target]
    colors = ScalarMappable(norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), cmap=ue_cmap)
    ax.set_theta_offset(0);ax.set_theta_direction(1)
    ax.bar(theta,values*scale,bottom=1,width=2*np.pi/n*.86,color=colors.to_rgba(values),edgecolor='white',lw=.25)
    theta_line=np.linspace(0,2*np.pi,361)
    ax.plot(theta_line,np.ones_like(theta_line),color='#555555',ls='--',lw=.6)
    for angle,value,name in zip(theta,values,d.Country):
        degrees=np.degrees(angle); flip=90<degrees<270
        ax.text(angle,max(1,1+value*scale)+.055,short_names.get(name,name),
            rotation=degrees+180 if flip else degrees,ha='right' if flip else 'left',
            va='center',fontsize=5.6,rotation_mode='anchor',clip_on=False)
    ax.set_ylim(0,2.03);ax.set_axis_off()
    ax.text(.5,.5,target,transform=ax.transAxes,ha='center',va='center',fontweight='bold',fontsize=10)

    if cax is None:
        cax = ax.inset_axes([0.12, -0.145, 0.76, 0.032], transform=ax.transAxes)
    ticks = np.linspace(-limit, limit, 5)
    cb = ax.figure.colorbar(colors, cax=cax, orientation='horizontal', extend='both',
                            ticks=ticks, format='%g')
    cb.ax.tick_params(length=2, labelsize=6, pad=2)
    cb.outline.set_linewidth(0.5)
    metric = 'frequency' if target == 'Frequency' else "MBEP"
    cb.set_label(f'Mean {metric} UE (pp yr$^{{-1}}$)', fontsize=6.5, labelpad=2)
    return cb


def scatter_size(n):
    return np.interp(np.log1p(n),[np.log1p(country.N_lakes.min()),np.log1p(country.N_lakes.max())],[10,70])


def draw_scatter(ax,legend=True):
    for group in [*GDP_GROUPS,'Unclassified']:
        d=country.loc[country.Country_GDP_group.eq(group)].sort_values('N_lakes',ascending=False)
        if len(d): ax.scatter(d.Frequency,d.MBEP,s=scatter_size(d.N_lakes),color=GROUP_COLORS[group],
            alpha=.8,edgecolors='white',lw=.35,label=group,zorder=3)
    x=np.array([country.Frequency.min(),country.Frequency.max()])
    ax.plot(x,slope*x+intercept,color='#555555',lw=.8,zorder=2)
    ax.axhline(0,color='#AAAAAA',ls='--',lw=.6,zorder=1)
    ax.axvline(0,color='#AAAAAA',ls='--',lw=.6,zorder=1)
    ax.margins(x=.12,y=.14)
    ax.set_xlabel('UE on frequency (pp yr$^{-1}$)')
    ax.set_ylabel('UE on MBEP (pp yr$^{-1}$)',labelpad=2)
    ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(5))
    p_text='$P$ < 0.001' if p_value<.001 else f'$P$ = {p_value:.3f}'
    ax.text(.04,.97,f'$r$ = {correlation:.2f}\n{p_text}',
        transform=ax.transAxes,va='top',fontsize=6)
    if legend:
        ax.legend(handles=[Line2D([],[],marker='o',ls='',color=GROUP_COLORS[g],label=g,markersize=4)
            for g in [*GDP_GROUPS,'Unclassified'] if country.Country_GDP_group.eq(g).any()],
            title='Country GDP growth',title_fontsize=5.7,loc='lower right',fontsize=5.7,
            labelspacing=.25,handletextpad=.25)


# 7. Assemble Figure 3

fig=plt.figure(figsize=(9.4,8.6))
top=.95
map_left,map_width=.025,.48
map_height=map_width*fig.get_figwidth()/fig.get_figheight()*np.ptp(outline[:,1])/np.ptp(outline[:,0])
ax_a=fig.add_axes([map_left,top-map_height,map_width,map_height])
ax_b=fig.add_axes([map_left,.632-map_height,map_width,map_height])
cax_a=fig.add_axes([map_left+.105,top-map_height-.026,.27,.008])
cax_b=fig.add_axes([map_left+.105,.632-map_height-.026,.27,.008])
draw_map(ax_a,'Frequency',show_legend=True,cax=cax_a)
draw_map(ax_b,'MBEP',show_legend=False,cax=cax_b)
label_panel(ax_a,'a',x=-.025,y=1.02);label_panel(ax_b,'b',x=-.025,y=1.02)

ax_c=fig.add_axes([.585,.585,.155,top-.585])
ax_d=fig.add_axes([.825,.585,.155,top-.585])
ax_c_strength=fig.add_axes([.585,.385,.155,.155])
ax_d_strength=fig.add_axes([.825,.385,.155,.155])
draw_column(ax_c,ax_c_strength,'Frequency')
draw_column(ax_d,ax_d_strength,'MBEP')
for ax,letter in [(ax_c,'c'),(ax_d,'d')]:
    label_panel(ax,letter,x=-.25,y=.02*map_height/ax.get_position().height+1)
fig.legend(handles=summary_handles,loc='center',bbox_to_anchor=(.79,.559),ncol=2,
    columnspacing=1.8,handlelength=1.2,handletextpad=.5,fontsize=6)

fig.canvas.draw()
renderer=fig.canvas.get_renderer()
target_bottom=cax_b.get_tightbbox(renderer).y0
for ax in [ax_c_strength,ax_d_strength]:
    shift=(target_bottom-ax.get_tightbbox(renderer).y0)/fig.bbox.height
    p=ax.get_position()
    ax.set_position([p.x0,p.y0+shift,p.width,p.height-shift])

row_top, row_base = .286, .036
rank_height = .250
ax_e = fig.add_axes([.005, row_top-rank_height, .31, rank_height], projection='polar')
ax_f = fig.add_axes([.325, row_top-rank_height, .31, rank_height], projection='polar')
ax_g = fig.add_axes([.735, row_base, .24, row_top-row_base])
draw_scatter(ax_g)

for ax, target in [(ax_e, 'Frequency'), (ax_f, 'MBEP')]:
    box = ax.get_position()
    key_width = .76 * box.width
    cax = fig.add_axes([box.x0 + (box.width-key_width)/2, row_base, key_width, .008])
    cb = draw_radial(ax, target, cax=cax)
    cb.ax.tick_params(axis='x', length=plt.rcParams['xtick.major.size'],
                      pad=plt.rcParams['xtick.major.pad'])
    cb.ax.xaxis.labelpad = ax_g.xaxis.labelpad
    cb.ax.xaxis.label.set_fontsize(ax_g.xaxis.label.get_fontsize())

for ax, letter, label_x in [(ax_e, 'e', -.02), (ax_f, 'f', -.02), (ax_g, 'g', -.25)]:
    box = ax.get_position()
    fig.text(box.x0 + label_x*box.width, row_top+.009, letter,
             va='bottom', fontweight='bold', fontsize=12)

plt.show()




