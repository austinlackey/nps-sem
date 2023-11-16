#!/usr/bin/env python
# coding: utf-8

# # SEM TEXT ANALYSIS

# ### Import Packages

# In[5]:


get_ipython().run_line_magic('run', 'cluster.py')


# ### Import Data

# In[6]:


# Importing Ref DataFrame
data_ref1 = pd.read_excel(r'..\Data\nationalDataDict.xlsx', sheet_name='Dictionary', usecols='A:D')
# data_ref1 = pd.read_excel(r'..\Data\NEW UPDATED DOWNLOADED DATA\National_SEM_Dictionary.xlsx', sheet_name='Dictionary', usecols='A:D') # NEW DATA
data_ref1.columns = [c.lower().replace(' ', '_') for c in data_ref1.columns]
print("Imported Ref Data.")

# Import Raw Data
data_raw = pd.read_parquet(r"..\Data\nationalDataRaw.parquet")
# data_raw = pd.read_excel(r"..\Data\NEW UPDATED DOWNLOADED DATA\2022_SEM_National.xlsx") # NEW DATA
print("Imported Raw Data.")
print('Raw Data Before Filter: ', data_raw.shape)

# Clean Data
data_clean = clean_data(data_raw.copy())
print('Clean Data After Filter: ', data_clean.shape)
data_clean.head()

# Other Frames
coded_data = pd.read_parquet(r'..\Data\codedData.parquet')
geodesic_park_information = pd.read_excel(r'..\Data\parkInformationGeodesic.xlsx', index_col=0)


# In[23]:


len([x for x in coded_data.columns if x.startswith('s_')])
len(coded_data.columns)


# In[24]:


coded_data.head()


# In[25]:


for col in coded_data.columns:
    print('\'', col, '\', ', sep='')


# ### Update Data

# In[26]:


# update_geodesic_park_data() # WARNING THIS TAKES A FEW MINUTES TO RUN
# code_data(data_clean, data_ref1, save=True) # WARNING THIS TAKES A FEW MINUTES TO RUN


# ### Un-pivot Data

# In[27]:


xbi_info = pd.melt(coded_data, id_vars=['n_IQualtricsID', 'weight_peak'], value_vars=['m_info_none', 
                    'm_info_previous', 
                    'm_info_friends', 
                    'm_info_call', 
                    'm_info_site', 
                    'm_info_osite', 
                    'm_info_local', 
                    'm_info_maps', 
                    'm_info_news', 
                    'm_info_units', 
                    'm_info_school', 
                    'm_info_social', 
                    'm_info_center', 
                    'm_info_tele', 
                    'm_info_ota', 
                    'm_info_book', 
                    'm_info_cruise', 
                    'm_info_grew', 
                    'm_info_other'], var_name='xbi_info', value_name='xbi_info_value')
xbi_info.dropna(subset=['xbi_info_value'], inplace=True)
xbi_info = xbi_info.replace({'m_info_none': 'Did not obtain infomation prior to visit', 
                    'm_info_previous': 'Previous visits', 
                    'm_info_friends': 'Friends/relatives/word of mouth', 
                    'm_info_call': 'Inquiry to park via phone/mail/email', 
                    'm_info_site': 'Official Park website', 
                    'm_info_osite': 'Other website', 
                    'm_info_local': 'Local businesses', 
                    'm_info_maps': 'Maps/brochures', 
                    'm_info_news': 'Newspaper/magazine articles', 
                    'm_info_units': 'Other units of the National Park System', 
                    'm_info_school': 'School class/program', 
                    'm_info_social': 'Social media', 
                    'm_info_center': 'State welcome center/visitor bureaus/chamber of commerce', 
                    'm_info_tele': 'Television/radio programs/DVDs', 
                    'm_info_ota': 'Online travel agent', 
                    'm_info_book': 'Travel guides/tour books', 
                    'm_info_cruise': 'Cruise ship', 
                    'm_info_grew': 'Live here/grew up here', 
                    'm_info_other': 'Other',
                    'Not Selected': 0,
                    'Selected': 1})
xbi_info.reset_index(drop=True)
xbi_info['xbi_info_value'] = np.where(xbi_info['xbi_info_value'] == 1, xbi_info['weight_peak'], np.float64(0))

xbi_prog = pd.melt(coded_data, id_vars=['n_IQualtricsID', 'weight_peak'], value_vars=['m_prog_rangerled', 
                    'm_prog_talkrang', 
                    'm_prog_outdoorex', 
                    'm_prog_indoorex', 
                    'm_prog_demo', 
                    'm_prog_brochure', 
                    'm_prog_center', 
                    'm_prog_movies', 
                    'm_prog_junior', 
                    'm_prog_jrself', 
                    'm_prog_stamp', 
                    'm_prog_audio', 
                    'm_prog_app', 
                    'm_prog_other'], var_name='xbi_prog', value_name='xbi_prog_value')
xbi_prog.dropna(subset=['xbi_prog_value'], inplace=True)
xbi_prog = xbi_prog.replace({'m_prog_rangerled': 'Attending a ranger-led activity, such as a tour or talk', 
                    'm_prog_talkrang': 'Talking informally with a ranger', 
                    'm_prog_outdoorex': 'Viewing outdoor exhibits', 
                    'm_prog_indoorex': 'Viewing indoor exhibits', 
                    'm_prog_demo': 'Attending a cultural demonstration or performance', 
                    'm_prog_brochure': 'Reading the park brochure or newspaper', 
                    'm_prog_center': 'Going to a visitor center', 
                    'm_prog_movies': 'Watching movies or videos about the park', 
                    'm_prog_junior': 'Participating with a child in your group in the Junior Ranger program', 
                    'm_prog_jrself': 'Participating in the Junior Ranger program', 
                    'm_prog_stamp': 'Obtaining a National Park passport stamp', 
                    'm_prog_audio': 'Listening to an audio tour or podcast', 
                    'm_prog_app': 'Using the National Park Service App', 
                    'm_prog_other': 'Other',
                    'Not Selected': 0,
                    'Selected': 1})
xbi_prog.reset_index(drop=True)
xbi_prog['xbi_prog_value'] = np.where(xbi_prog['xbi_prog_value'] == 1, xbi_prog['weight_peak'], np.float64(0))

xbi_age = pd.melt(coded_data, id_vars=['n_IQualtricsID', 'weight_peak'], value_vars=['n_adage_1', 
                    'n_adage_2', 
                    'n_adage_3', 
                    'n_adage_4', 
                    'n_adage_5', 
                    'n_adage_6', 
                    'n_cage_1', 
                    'n_cage_2', 
                    'n_cage_3', 
                    'n_cage_4', 
                    'n_cage_5', 
                    'n_cage_6'], var_name='xbi_age', value_name='xbi_age_value')
xbi_age.dropna(subset=['xbi_age_value'], inplace=True)
xbi_age['xbi_age_group'] = xbi_age['xbi_age_value'].apply(slice_age_groups)
xbi_age['xbi_age_group_pos'] = xbi_age['xbi_age_group'].apply(add_group_positions, positions=age_group_positions)

xbi_visitor_age = coded_data[['n_IQualtricsID', 'c_years']].copy()
xbi_visitor_age.dropna(subset=['c_years'], inplace=True)
xbi_visitor_age['xbi_age_group_pos'] = xbi_visitor_age['c_years'].apply(add_group_positions, positions=visitor_age_group_positions)
#r_motiv columns
xbi_motiv = pd.melt(coded_data, id_vars='n_IQualtricsID', value_vars=['r_motiv_visit', 
                    'r_motiv_solitude', 
                    'r_motiv_sounds', 
                    'r_motiv_learn', 
                    'r_motiv_history', 
                    'r_motiv_family', 
                    'r_motiv_exercise', 
                    'r_motiv_wildlife', 
                    'r_motiv_relax', 
                    'r_motiv_stars', 
                    'r_motiv_other'], var_name='xbi_motiv', value_name='xbi_motiv_value')
xbi_motiv.dropna(subset=['xbi_motiv_value'], inplace=True)
xbi_motiv = xbi_motiv.replace({'r_motiv_visit': 'To visit a National Park Service site', 
                    'r_motiv_solitude': 'To experience solitude', 
                    'r_motiv_sounds': 'To hear the sounds of nature/quiet', 
                    'r_motiv_learn': 'To learn more about nature', 
                    'r_motiv_history': 'To learn more about American history and culture', 
                    'r_motiv_family': 'To spend time with family/friends', 
                    'r_motiv_exercise': 'To get physical exercise', 
                    'r_motiv_wildlife': 'To view wildlife or natural scenery', 
                    'r_motiv_relax': 'To relax', 
                    'r_motiv_stars': 'To view dark night sky/stars', 
                    'r_motiv_other': 'Other'})
#r_agree columns
xbi_agree = pd.melt(coded_data, id_vars='n_IQualtricsID', value_vars=['r_agree_safe', 
                    'r_agree_crowded', 
                    'r_agree_pristine', 
                    'r_agree_fee', 
                    'r_agree_crime', 
                    'r_agree_access', 
                    'r_agree_history', 
                    'r_agree_develop'], var_name='xbi_agree', value_name='xbi_agree_value')
xbi_agree.dropna(subset=['xbi_agree_value'], inplace=True)
xbi_agree = xbi_agree.replace({'r_agree_safe': 'Park is a safe place to visit', 
                    'r_agree_crowded': 'Park is too crowded', 
                    'r_agree_pristine': 'Natural resources in Park are in pristine condition', 
                    'r_agree_fee': 'The entrance fee for Park is too high', 
                    'r_agree_crime': 'Vandalism and crime are not a problem at Park', 
                    'r_agree_access': 'Park is not accessible to a person with physical disabilities', 
                    'r_agree_history': 'Historical and cultural features in Park are well maintained/preserved', 
                    'r_agree_develop': 'Development of adjacent areas detracts from visitors’ experience at Park'})
#r_qual columns
xbi_qual = pd.melt(coded_data, id_vars='n_IQualtricsID', value_vars=['r_qual_center', 
                    'r_qual_exhib', 
                    'r_qual_restroom', 
                    'r_qual_walkway', 
                    'r_qual_camp', 
                    'r_qual_emp', 
                    'r_qual_map', 
                    'r_qual_ranger', 
                    'r_qual_value', 
                    'r_qual_service', 
                    'r_qual_other', 
                    'r_qual_learn', 
                    'r_qual_rec', 
                    'r_quality'], var_name='xbi_qual', value_name='xbi_qual_value')
xbi_qual.dropna(subset=['xbi_qual_value'], inplace=True)
xbi_qual = xbi_qual.replace({'r_qual_center': 'Visitor Center', 
                    'r_qual_exhib': 'Exhibits (indoor/outdoor)', 
                    'r_qual_restroom': 'Restrooms', 
                    'r_qual_walkway': 'Walkways, trails, and roads', 
                    'r_qual_camp': 'Campgrounds and/or picnic areas', 
                    'r_qual_emp': 'Assistance from park employees', 
                    'r_qual_map': 'Park map or brochure', 
                    'r_qual_ranger': 'Ranger programs', 
                    'r_qual_value': 'Value for entrance fee paid', 
                    'r_qual_service': 'Commercial services in the park', 
                    'r_qual_other': 'Other services', 
                    'r_qual_learn': 'Learning about nature/history/culture', 
                    'r_qual_rec': 'Outdoor recreation', 
                    'r_quality': 'Overall park quality'})


# In[28]:


# save_pivot_data([xbi_info, xbi_age, xbi_visitor_age, xbi_motiv, xbi_agree, xbi_qual, xbi_prog], ['xbi_info', 'xbi_age', 'xbi_visitor_age', 'xbi_motiv', 'xbi_agree', 'xbi_qual', 'xbi_prog'])


# In[29]:


# MODEL TO USE FOR NLP
modelGN300 = api.load('word2vec-google-news-300')
# api.info()


# In[30]:


get_text_columns(data_clean)


# ## VAR1: o_least1

# In[31]:


# DIM_REDU: TSNE
o_least1_model1 = NLP(data_clean['o_least1'].iloc[:], embedding_model=modelGN300, model_name='o_least_model1')
o_least1_model1.dimension_reduce('tsne', perplexity=20)


# In[32]:


#CLUST: KMEANS
o_least1_model1.clusterize(cluster_algorithm='kmeans', num_clusters=25)
o_least1_model1.generate_cluster_graph(figsize=(10,10), num_annotations = 200, jitter_amount=0.01, hide_labels=False, hide_legend=False, save=False)


# ## VAR2: o_like1

# In[33]:


# DIM_REDU: TSNE
o_like_model1 = NLP(data_clean['o_like1'].iloc[:], embedding_model=modelGN300, model_name='o_like_model1')
o_like_model1.dimension_reduce('tsne', perplexity=20)


# In[34]:


#CLUST: KMEANS
o_like_model1.clusterize(cluster_algorithm='kmeans', num_clusters=30)
o_like_model1.generate_cluster_graph(figsize=(10,10), num_annotations = 200, jitter_amount=0.01, hide_labels=False, hide_legend=False)


# ## VAR3: o_subjects

# In[35]:


# DIM_REDU: TSNE
o_subjects_model1 = NLP(data_clean['o_subjects'].iloc[:], embedding_model=modelGN300, model_name='o_subjects_model1')
o_subjects_model1.dimension_reduce('tsne', perplexity=20)


# In[36]:


o_subjects_model1.datapoints


# In[37]:


#CLUST: KMEANS
o_subjects_model1.clusterize(cluster_algorithm='kmeans', num_clusters=25)
o_subjects_model1.generate_cluster_graph(figsize=(10,10), num_annotations = 200, max_char_length=20, jitter = False, jitter_amount=2, hide_labels=False, hide_legend=False)


# In[38]:


# DIM_REDU: TSNE
o_subjects_model2 = NLP(data_clean['o_subjects'].iloc[:], embedding_model=modelGN300, model_name='o_subjects_model2')
o_subjects_model2.dimension_reduce('spectral_emb', perplexity=20)


# In[39]:


#CLUST: KMEANS
o_subjects_model2.clusterize(cluster_algorithm='kmeans', num_clusters=25)
o_subjects_model2.generate_cluster_graph(figsize=(10,10), num_annotations = 200, max_char_length=20, jitter = False, jitter_amount=2, hide_labels=False, hide_legend=False)


# ## VAR4: o_addl

# In[40]:


# DIM_REDU: TSNE
o_addl_model1 = NLP(data_clean['o_addl'].iloc[:], embedding_model=modelGN300, model_name='o_addl_model1')
o_addl_model1.dimension_reduce('tsne', perplexity=50)


# In[41]:


#CLUST: KMEANS
o_addl_model1.clusterize(cluster_algorithm='kmeans', num_clusters=25)
o_addl_model1.generate_cluster_graph(figsize=(10,10), num_annotations = 200, max_char_length=40, jitter = False, jitter_amount=2, hide_labels=False, hide_legend=False)


# In[42]:


# DIM_REDU: TSNE
o_addl_model2 = NLP(data_clean['o_addl'].iloc[:], embedding_model=modelGN300, model_name='o_addl_model2')
o_addl_model2.dimension_reduce('spectral_emb', perplexity=50)


# In[43]:


#CLUST: KMEANS
o_addl_model2.clusterize(cluster_algorithm='kmeans', num_clusters=25)
o_addl_model2.generate_cluster_graph(figsize=(10,10), num_annotations = 200, max_char_length=40, jitter = False, jitter_amount=2, hide_labels=False, hide_legend=False)


# In[44]:


df_1 = o_addl_model1.get_cluster_info(4)
print(df_1.shape)
df_1


# # Interesting Cluster Notes

# cluster# (themes) #of_responses
# 
# ### o_least1_model1
# - 19 (crowds) 84
# - 5 (limited parking / crowding) 159
# - 14 (busy / traffic) 83
# - 18 (nothing to dislike) 93
# - 1 (covid measures / timed entry) 86
# - 15 (lack of cell service / wifi)  119
# - 6 (mosquitos / bugs) 40
# - 16 (didn't have enough time / loved everything) 175
# 
# ### o_like_model1
# - 8 (hiking) 127
# - 20 (hiking trails) 129
# - 25 (hikes / trails) 52
# - 13 (nature) 139
# - 11 (views) 101
# - 1 (memorials / architecture / statues) 212
# - 18 (tours / guides) 82
# - 21 (solitude / quietness) 134
# - 10 (visitor centers / exhibits / videos / museums) 175
# - 2 (natural beauty) 200
# - 3 (scenery) 346
# 
# ### o_subjects_model1
# - 12 (history) 41
# - 10 (american history / heritage) 66
# - 8 ('war related information') 77
# - 14 (wildlife) 29
# - 13 (geology) 34
# 
# ### o_addl_model1
# - 17 (thank you!) 32
# - 4 (no) 86
# - 20 ('survey related') 51
# - 1 ('positive feedback regarding survey work') 114
# - 6 (enjoyed visit) 94
#     

# ## Saving Data

# ### Models

# In[22]:


# 
# models_data = merge_model_data(models = [o_least1_model1, o_like_model1, o_subjects_model1, o_subjects_model2, o_addl_model1, o_addl_model2])

# save_data(models_data, 'models_data')


# # FOR DOCUMENTATION PURPOSES

# In[14]:


# data.loc[(data['cluster'] == 15) | (data['cluster'] == 6)].sort_values(by='cluster').head(100)
test = pd.DataFrame(o_least1_model1.vectorized_docs)
test2 = o_least1_model1.datapoints
print(test.shape, test2.shape)
test2 = test2[(test2['cluster'] == 1)].reset_index(drop=True)
test2 = test2.loc[0:100]
test = test.loc[test2.index, 0:9].reset_index(drop=True)
test2


from matplotlib.path import Path
import matplotlib.patches as patches
ynames = test.columns
ys = test
ymins = ys.min(axis=0)
ymaxs = ys.max(axis=0)
dys = ymaxs - ymins
ymins -= dys * 0.05  # add 5% padding below and above
ymaxs += dys * 0.05

# ymaxs[1], ymins[1] = ymins[1], ymaxs[1]  # reverse axis 1 to have less crossings
# dys = ymaxs - ymins

# transform all data to be compatible with the main axis
zs = np.zeros_like(ys)
zs[:, 0] = ys.iloc[:, 0]
zs[:, 1:] = (ys.iloc[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

fig, host = plt.subplots(figsize=(10,4))

axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
for i, ax in enumerate(axes):
    ax.set_ylim(ymins[i], ymaxs[i])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if ax != host:
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_ticks_position('right')
        ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

host.set_xlim(0, ys.shape[1] - 1)
host.set_xticks(range(ys.shape[1]))
host.set_xticklabels(ynames, fontsize=14)
host.tick_params(axis='x', which='major', pad=7)
host.spines['right'].set_visible(False)
host.xaxis.tick_top()
host.set_title('Parallel Coordinates Plot', fontsize=18, pad=12)

colors = [plt.cm.inferno(index) for index in np.linspace(0, 1, 5)]

legend_handles = [None for _ in np.arange(0, max(test2.cluster) + 1)]
for j in range(ys.shape[0]):
    # create bezier curves
    verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                     np.repeat(zs[j, :], 3)[1:-1]))
    codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', lw=2, alpha=0.4, edgecolor=colors[test2.cluster[j]])
    legend_handles[test2.cluster.to_numpy()[j]] = patch
    host.add_patch(patch)
# host.legend(legend_handles, np.unique(test2.cluster.astype('<U10')),
#             loc='lower center', bbox_to_anchor=(0.5, -0.18),
#             ncol=len(test2.cluster), fancybox=True, shadow=True)
plt.tight_layout()
plt.show()
fig.savefig(r'..\Writeup Files/parallel_coord_plot.png', transparent=True)


# In[8]:


# test_data = pd.Series(['like want'])
test_data = pd.Series('hate want'.split())
test = NLP(test_data, embedding_model=modelGN300, model_name='test')
test.tokenized_docs
vecs = pd.DataFrame(test.vectorized_docs)
plt.figure(figsize=(10, 3), dpi=100)
sns.heatmap(vecs[vecs.columns[:]], linewidth=0, cmap='magma')


# In[13]:


o_least1_model1.generate_cluster_graph(figsize=(10,10), num_annotations = 200, jitter_amount=0.01, hide_labels=False, hide_legend=False)
data = o_least1_model1.datapoints


# In[23]:


fig = plt.figure()
plt.scatter(data['component1'], data['component2'], color='black', alpha=0.3)
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('TSNE Reduced Dimensions')
plt.show()
fig.savefig(r'..\Writeup Files/reduced_dim.png', transparent=True)


# In[7]:


data_sorted = data.loc[(data['cluster'] == 15) | (data['cluster'] == 6)].sort_values(by='cluster').head(100)
print(data_sorted.shape)
data_sorted.head(50)


# In[8]:


from scipy.ndimage.filters import gaussian_filter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

start = 1800
end = 1850
length = end - start

def curveF(x1, x2):
       return 0.5 * x1 + 0.6 * x2 + 0.2 * x1 * x1 + 0.1 * x1 * x2 + 0.3 * x2 * x2 + 4

vecs = pd.DataFrame(o_least1_model1.vectorized_docs)
vecs = vecs.loc[data_sorted.index]
# vecs = vecs.iloc[start:end][:]

X = np.arange(vecs.shape[0])
Y = np.arange(vecs.shape[1])
X, Y = np.meshgrid(X, Y)

vecs = vecs.T.apply(lambda row: gaussian_filter(row, sigma=20)).T
vecs = pd.DataFrame(gaussian_filter(vecs, sigma=2))
# Z = pd.DataFrame(np.zeros((vecs.shape[0], vecs.shape[1]))).to_numpy().T
Z = vecs.to_numpy().T
# xz = np.linspace(-5, 5, vecs.shape[0])
# yz = np.linspace(-5, 5, vecs.shape[1])
# xz, yz = np.meshgrid(xz, yz)
# amplitude = 0.01
# Z = amplitude * np.sin(np.sqrt(xz**2 + yz**2))
# Z = curveF(xz, yz)

C = vecs.to_numpy().T
scamap = plt.cm.ScalarMappable(cmap='inferno')
fcolors = scamap.to_rgba(C)
rcount, ccount, _ = fcolors.shape

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(X, Y, Z, cmap='inferno', antialiased = True, alpha=1, rstride=5, cstride=5, edgecolor="black", linewidth=0.5)

surf = ax.plot_surface(X, Y, Z, facecolors=fcolors, shade=False, rcount=rcount, ccount=ccount)
surf.set_facecolor((0,0,0,0))

# ax.plot_surface(X, Y, Z, facecolors=fcolors, antialiased = True, rstride=5, cstride=5)
# ax.plot_wireframe(X, Y, Z, color ='black', rstride=10, cstride=10)
# ax.plot_surface(X, Y, Z, facecolors=fcolors, antialiased = True)
# ax.plot_wireframe(X, Y, Z, color ='black')

fig.set_figwidth(18)
fig.set_figheight(18)

ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])
ax.view_init(35, -40)
ax.set_axis_off()
ax.set_facecolor((1, 1, 1))

# Create the scatter plot
surf = ax.plot_surface(X, Y, Z, facecolors=fcolors, shade=False, rcount=rcount, ccount=ccount)
surf.set_facecolor((0,0,0,0))

# Calculate the min and max values of the plot
plot_min = np.min(Z)
plot_max = np.max(Z)

# Add a color bar as a legend
scamap.set_array([])
cbar = plt.colorbar(scamap, ax=ax, shrink=0.5, orientation='vertical', pad=0.1, aspect=20, ticks=[plot_min, plot_max])
cbar.set_label('Color Legend')
cbar.ax.set_yticklabels([f'{plot_min:.2f}', f'{plot_max:.2f}'])

plt.show()
fig.savefig(r'..\Writeup Files/clothTrans.png', transparent=True)


# In[387]:


vecs = pd.DataFrame(o_least1_model1.vectorized_docs)
vecs = vecs.loc[data_sorted.index]

plt.figure(figsize=(10, 3), dpi=100)
sns.heatmap(vecs.loc[[2966, 3014]], linewidth=0, cmap='magma')

