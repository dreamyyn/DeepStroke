import nibabel as nib
import numpy as np

list_nonreper_test1 = ["03032","01020","01002","05002","08009","03045","11003","01042","30058A","30082A","08010","30049A","30073A"]
list_nonreper_test2 = ["09005","05011","05010","03002","01021","10002","30008A","30054","01010","30030A","01047","30032","03028"]
list_nonreper_test3 = ["09007","05006","11004","01017","03040","01007","05008","30007","01036","09006","09002","30043","10001"]
list_nonreper_test4 = ["03025","30037","09003","05007","05012","05003","30027A","05005","11002","10006","10007","30084A","02003"]
list_nonreper_test5 = ["01027","01038","01041","02005","01040","08007","10004","03018","09004","01004","01045",'10005','11001']
list_reper_test1 = ["30012","01028","30042","30116","03027","30006A","03017","30068","03043","03016","01003","03046","30099","30117","30046A","30102","03042","30028","03047","01029","01006"]
list_reper_test2 = ["08005","01048","30063","30101","05001","10003","03008","30069A","08008","30035A","03007","30041","30040","30096","30126","08003","30108","30097","03033","30120","01001"]
# 30018A belongs to test 3, but with no GRE. so temperarily removed. if tested in any combination with no GRE, should include 30018A.
list_reper_test3 = ["30018A","03036","30122","30048A","30053","03039","30098","30055","03026","01015","03037","30127","30078A","30071A","03024","01043","03011","30057A","30024A","30115","30002A"]
list_reper_test4 = ["03013","03035","08001","03031","30056","02004","03041","01032","03019","30103","30023","30061","03009","03020","30047A","30026A","30109","05009","03048","10009","30090A"]
list_reper_test5 = ["30092","30022A","30034","30106","30080","03001","01049","12001","30113","01044","30075A","30059A","03003","30077A","30045A","30124","02006","30001A"]
subj_list_penumbra = sorted(list_nonreper_test1)+sorted(list_nonreper_test2)+sorted(list_nonreper_test3)+sorted(list_nonreper_test4)+sorted(list_nonreper_test5)
subj_list_core = sorted(list_reper_test1)+sorted(list_reper_test2)+sorted(list_reper_test3)+sorted(list_reper_test4)+sorted(list_reper_test5)

neg_sample = 0
pos_sample = 0
for subject_name in subj_list_penumbra + subj_list_core:
  lesion_path = '/Users/admin/deepstroke173/PWImasked173/' + subject_name + '/LESION.nii'
  lesion_load = nib.load(lesion_path)
  lesion = (lesion_load.get_fdata() > 0.9) * 1.
  lesion = np.maximum(0, np.nan_to_num(lesion, 0))
  brain_mask_data = nib.load('/Users/admin/controls_stroke_DL/001/T1.nii')
  brain_mask = (brain_mask_data.get_fdata() > 0) * 1.
  brain_mask[np.isnan(brain_mask)] = np.NaN
  lesion_f = np.array(lesion).flatten()
  brain_mask_f = np.array(brain_mask[:,:,13:73]).flatten()
  masked_lesion = lesion_f * brain_mask_f
  result = masked_lesion[~np.isnan(masked_lesion)]
  pos_sample += np.sum(result)
  neg_sample += (len(result) - np.sum(result))
  print(np.sum(result), len(result))
print(neg_sample, pos_sample)
# neg sample is 76 times more than pos samples