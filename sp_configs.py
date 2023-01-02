import json
from copy import deepcopy


pdfdict = {"lanet": "https://openaccess.thecvf.com/content/ACCV2022/papers/Wang_Rethinking_Low-level_Features_for_Interest_Point_Detection_and_Description_ACCV_2022_paper.pdf",
            "kp2d": "https://openreview.net/pdf?id=Skx82ySYPH",
            "superpoint_indep": "https://arxiv.org/abs/2007.15122",
            "superpoint": "https://arxiv.org/abs/1712.07629",
            }
repo_dict = {"lanet": "https://github.com/wangch-g/lanet",
            "kp2d":"https://github.com/TRI-ML/KP2D",
            "superpoint": "https://github.com/magicleap/SuperPointPretrainedNetwork",
             "superpoint_indep": "https://github.com/eric-yyjau/pytorch-superpoint",
             }
method_name_dict = {"lanet": "LANet",
                    "kp2d": "KeypointNet",
                    "superpoint_indep": "DeepFEPE SuperPoint",
                    "superpoint": "MagicLeap SuperPoint"}

metadata_dict =  {
    "publish_anonymously": False,
    "authors": "Submitted by benchmark orgs, implementation by paper authors",
    "contact_email": "ducha.aiki@gmail.com",
    "method_name": "",
    "method_description":
    r"""Baseline, PUT 2048 features
    Matched using the built-in matcher (bidirectional filter with the 'both' strategy,
    hopefully optimal inlier and ratio test thresholds) with DEGENSAC""",
    "link_to_website": "",
    "link_to_pdf": ""
}

config_common_dict =  {"json_label": "rootsiftfix",
    "keypoint": "korniadogfix",
    "descriptor": "rootsift",
    "num_keypoints": 2048}


matcher_template_dict = {
     "method": "nn",
     "distance": "L2",
     "flann": False,
     "num_nn": 1,
     "filtering": {
         "type": "snn_ratio_pairwise",
         "threshold": 0.99,
     },
     "symmetric": {
         "enabled": True,
         "reduce": "both",
     }
}

geom_template_dict =  {"method": "cmp-degensac-f",
                "threshold": 0.75,
                "confidence": 0.999999,
                "max_iter": 100000,
                "error_type": "sampson",
                "degeneracy_check": True,
            }

base_config =  {
    "metadata": metadata_dict,
    "config_common": config_common_dict,
    "config_phototourism_stereo": {
        "use_custom_matches": False,
        "matcher": deepcopy(matcher_template_dict),
        "outlier_filter": { "method": "none" },
        "geom": deepcopy(geom_template_dict)
        },
    "config_pragueparks_stereo": {
        "use_custom_matches": False,
        "matcher": deepcopy(matcher_template_dict),
        "outlier_filter": { "method": "none" },
        "geom": deepcopy(geom_template_dict)
        },

}

if __name__ == '__main__':
    kp = 'kp2d'
    for v in ['v0','v1', 'v2', 'v3', 'v4']:
        for norm in ['', '_norm']:
            json_fn = f'{kp}_{v}{norm}_1024'
            json_fn2 = f'{kp}_{v}{norm}_1024'.replace('_','-')
            md = deepcopy(metadata_dict)
            md['link_to_website'] = repo_dict[kp]
            md['link_to_pdf'] = pdfdict[kp]
            md['method_name'] = method_name_dict[kp]
            md['method_description']=md['method_description'].replace('PUT', method_name_dict[kp])
            common =  {"json_label": json_fn2,
                "keypoint": json_fn2,
                "descriptor": json_fn2,
                "num_keypoints": 2048}
            base_config =  {
                "metadata": md,
                "config_common": common,
                "config_phototourism_stereo": {
                    "use_custom_matches": False,
                    "matcher": deepcopy(matcher_template_dict),
                    "outlier_filter": { "method": "none" },
                    "geom": deepcopy(geom_template_dict)
                    },
                "config_pragueparks_stereo": {
                    "use_custom_matches": False,
                    "matcher": deepcopy(matcher_template_dict),
                    "outlier_filter": { "method": "none" },
                    "geom": deepcopy(geom_template_dict)
                    }
            }
            with open(f'OUT_JSON/{json_fn}.json', 'w') as f:
                json.dump([base_config], f, indent=2)
            match_ths = [0.85, 0.9, 0.95, 0.99]
            inl_ths = [0.5, 0.75, 1.0, 1.25]
            configs = []
            for match_th in match_ths:
                for inl_th in inl_ths:
                    current_config = deepcopy(base_config)
                    for dset in ['phototourism', 'pragueparks']:
                        current_config[f'config_{dset}_stereo']['geom']['threshold'] = inl_th
                        current_config[f'config_{dset}_stereo']['matcher']['filtering']['threshold'] = match_th
                    label = current_config['config_common']['json_label']
                    current_config['config_common']['json_label']  = f'{label}-snnth-{match_th}-inlth-{inl_th}'
                    configs.append(current_config)
            with open(f'OUT_JSON_RANSAC/{json_fn}_tuning.json', 'w') as f:
                json.dump(configs, f, indent=2)
    kp = 'superpoint'
    for v in ['_magicleap']:
        for norm in ['', '_norm']:
            json_fn = f'{kp}{v}{norm}_1024'
            json_fn2 = f'{kp}{v}{norm}_1024'.replace('_','-')
            md = deepcopy(metadata_dict)
            md['link_to_website'] = repo_dict[kp]
            md['link_to_pdf'] = pdfdict[kp]
            md['method_name'] = method_name_dict[kp]
            md['method_description']=md['method_description'].replace('PUT', method_name_dict[kp])
            common =  {"json_label": json_fn2,
                "keypoint": json_fn2,
                "descriptor": json_fn2,
                "num_keypoints": 2048}
            base_config =  {
                "metadata": md,
                "config_common": common,
                "config_phototourism_stereo": {
                    "use_custom_matches": False,
                    "matcher": deepcopy(matcher_template_dict),
                    "outlier_filter": { "method": "none" },
                    "geom": deepcopy(geom_template_dict)
                    },
                "config_pragueparks_stereo": {
                    "use_custom_matches": False,
                    "matcher": deepcopy(matcher_template_dict),
                    "outlier_filter": { "method": "none" },
                    "geom": deepcopy(geom_template_dict)
                    }
            }
            with open(f'OUT_JSON/{json_fn}.json', 'w') as f:
                json.dump([base_config], f, indent=2)
            match_ths = [0.85, 0.9, 0.95, 0.99]
            inl_ths = [0.5, 0.75, 1.0, 1.25, 1.5]
            configs = []
            for match_th in match_ths:
                for inl_th in inl_ths:
                    current_config = deepcopy(base_config)
                    for dset in ['phototourism', 'pragueparks']:
                        current_config[f'config_{dset}_stereo']['geom']['threshold'] = inl_th
                        current_config[f'config_{dset}_stereo']['matcher']['filtering']['threshold'] = match_th
                    label = current_config['config_common']['json_label']
                    current_config['config_common']['json_label']  = f'{label}-snnth-{match_th}-inlth-{inl_th}'
                    configs.append(current_config)
            with open(f'OUT_JSON_RANSAC/{json_fn}_tuning.json', 'w') as f:
                json.dump(configs, f, indent=2)
    kp = 'lanet'
    for v in ['v0','v1']:
        for norm in ['', '_norm']:
            json_fn = f'{kp}_{v}{norm}_1024'
            json_fn2 = f'{kp}_{v}{norm}_1024'.replace('_','-')
            md = deepcopy(metadata_dict)
            md['link_to_website'] = repo_dict[kp]
            md['link_to_pdf'] = pdfdict[kp]
            md['method_name'] = method_name_dict[kp]
            md['method_description']=md['method_description'].replace('PUT', method_name_dict[kp])
            common =  {"json_label": json_fn2,
                "keypoint": json_fn2,
                "descriptor": json_fn2,
                "num_keypoints": 2048}
            base_config =  {
                "metadata": md,
                "config_common": common,
                "config_phototourism_stereo": {
                    "use_custom_matches": False,
                    "matcher": deepcopy(matcher_template_dict),
                    "outlier_filter": { "method": "none" },
                    "geom": deepcopy(geom_template_dict)
                    },
                "config_pragueparks_stereo": {
                    "use_custom_matches": False,
                    "matcher": deepcopy(matcher_template_dict),
                    "outlier_filter": { "method": "none" },
                    "geom": deepcopy(geom_template_dict)
                    }
            }
            with open(f'OUT_JSON/{json_fn}.json', 'w') as f:
                json.dump([base_config], f, indent=2)
            match_ths = [0.85, 0.9, 0.95, 0.99]
            inl_ths = [0.5, 0.75, 1.0, 1.25]
            configs = []
            for match_th in match_ths:
                for inl_th in inl_ths:
                    current_config = deepcopy(base_config)
                    for dset in ['phototourism', 'pragueparks']:
                        current_config[f'config_{dset}_stereo']['geom']['threshold'] = inl_th
                        current_config[f'config_{dset}_stereo']['matcher']['filtering']['threshold'] = match_th
                    label = current_config['config_common']['json_label']
                    current_config['config_common']['json_label']  = f'{label}-snnth-{match_th}-inlth-{inl_th}'
                    configs.append(current_config)
            with open(f'OUT_JSON_RANSAC/{json_fn}_tuning.json', 'w') as f:
                json.dump(configs, f, indent=2)
    kp = 'superpoint_indep'
    for v in ['coco', 'kitty']:
        for norm in ['', '_norm']:
            for sp in ['', '_subpix']:
                json_fn = f'{kp}_{v}{norm}_1024{sp}'
                json_fn2 = f'{kp}_{v}{norm}_1024{sp}'.replace('_','-')
                md = deepcopy(metadata_dict)
                md['link_to_website'] = repo_dict[kp]
                md['link_to_pdf'] = pdfdict[kp]
                md['method_name'] = method_name_dict[kp]
                md['method_description']=md['method_description'].replace('PUT', method_name_dict[kp])
                common =  {"json_label": json_fn2,
                    "keypoint": json_fn2,
                    "descriptor": json_fn2,
                    "num_keypoints": 2048}
                base_config =  {
                    "metadata": md,
                    "config_common": common,
                    "config_phototourism_stereo": {
                        "use_custom_matches": False,
                        "matcher": deepcopy(matcher_template_dict),
                        "outlier_filter": { "method": "none" },
                        "geom": deepcopy(geom_template_dict)
                        },
                    "config_pragueparks_stereo": {
                        "use_custom_matches": False,
                        "matcher": deepcopy(matcher_template_dict),
                        "outlier_filter": { "method": "none" },
                        "geom": deepcopy(geom_template_dict)
                        }
                }
                with open(f'OUT_JSON/{json_fn}.json', 'w') as f:
                    json.dump([base_config], f, indent=2)
                match_ths = [0.85, 0.9, 0.95, 0.99]
                inl_ths = [0.5, 0.75, 1.0, 1.25]
                configs = []
                for match_th in match_ths:
                    for inl_th in inl_ths:
                        current_config = deepcopy(base_config)
                        for dset in ['phototourism', 'pragueparks']:
                            current_config[f'config_{dset}_stereo']['geom']['threshold'] = inl_th
                            current_config[f'config_{dset}_stereo']['matcher']['filtering']['threshold'] = match_th
                        label = current_config['config_common']['json_label']
                        current_config['config_common']['json_label']  = f'{label}-snnth-{match_th}-inlth-{inl_th}'
                        configs.append(current_config)
                with open(f'OUT_JSON_RANSAC/{json_fn}_tuning.json', 'w') as f:
                    json.dump(configs, f, indent=2)

