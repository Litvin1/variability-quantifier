import image_viewer
import beehive
import pandas as pd
'''
images_to_show = {
    'phase': {'hyb': 0, 'ch': 'phase', 'cmap': 'gray'},
    #'dapi' : {'hyb' : 0, 'ch' : 'dapi', 'cmap' : 'cyan','limits' : (2000,15000)},
     #'lasI' : {'hyb' : 10,'ch' : 'A647', 'cmap' : 'red', 'limits' : (500,3500)}, #R24 algU ; muchA R58, lasI R10
     #'algU' : {'hyb' : 32,'ch' : 'A550', 'cmap' : 'green', 'limits' : (500,2000)},
     'mucA' : {'hyb' : 37,'ch' : 'A647', 'cmap' : 'cyan', 'limits' : (500,1500)},
    'lysC': {'hyb': 36, 'ch': 'A488', 'cmap': 'magenta', 'limits': (500, 1500)},  # R53
     #'dnaA' : {'hyb' : 16,'ch' : 'A488', 'cmap' : 'white', 'limits' : (500,1500)}, #R53

}
'''
# for simple viewer
fov = 0
spot_threshold = 0
# old exp:
main_dir = r'Z:\danielll\analysis\dl_auto_011123'
# new exp:
#main_dir = r'Z:\danielll\analysis\dl_auto_270224_march_24_analysis'

#for beehive

cbd_path = rf'{main_dir}/cell_by_gene/merged_cell_by_gene.norm.filt.txt'
summary_path = rf'{main_dir}/automation_design/automation_summary.xlsx'
gene_to_job_path = rf'{main_dir}/automation_design/readout_to_gene_to_job.xlsx'
#P2F gene name extractor
cell_by_gene_df = beehive.get_cell_by_gene(cbd_path, exclude_unclassified=True)

'''
viewer = image_viewer.viewer(main_dir=main_dir,
                             interpolation='nearest',
                             is_show_seg=True,
                             is_show_spots=False,
                             is_cell_spots_only=False,
                             spot_threshold=spot_threshold,
                             raw_data_dir=r'Z:\imaging_data\zp_auto_290224\raw_data',  # for loading 3D images
                             show_demultiplexing=None,
                             # 'genes', #'samples' #'samples' # None | 'genes' #genes | samples
                             shrink_mask=False
                             )

viewer.show_image(images_to_show=images_to_show, fov=fov, spot_size=1)
'''

num_of_cells = 800
gene_name = 'ubiE'
OD = '0.5'
panel_size_mult = 2

df = cell_by_gene_df[cell_by_gene_df['sample_name'].str.split('_').str[-1] == OD]
#df = cell_by_gene_df

### view
genes_to_view = {
    # 'mucA' : { 'color':'red','contrast_limits' : (300,2000), 'disp_name' : 'mucA' },
    gene_name : { 'color':'green','contrast_limits' : (300,2000), 'disp_name' : gene_name },
    #'rpoH' : { 'color':'cyan','contrast_limits' : (300,2000), 'disp_name' : 'rpoH' },
}

selected_cbg_df = beehive.view_cells_cell_by_gene(main_dir,
                        summary_path = summary_path,
                        gene_to_job_path = gene_to_job_path,
                        cell_by_gene_df = df, #your cell by gene
                        num_of_cells = min(num_of_cells,len(df)),
                        genes_to_view= genes_to_view,
                        randomize = False,
                        sort_by=gene_name, # if you don't want to sort, write None or remove this line
                        crop_size=75,
                        panel_size=(1024*panel_size_mult, 1024*panel_size_mult),
                        k = 0,
                        dilate_by = 1,
                        # see only the cell without the cells next to it
                        turn_off_bg=True,
                        show_masks = True
                        )
