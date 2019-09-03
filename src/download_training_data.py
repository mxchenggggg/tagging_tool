from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

g_login = GoogleAuth()
g_login.LocalWebserverAuth()
drive = GoogleDrive(g_login)

# with open("./test.txt") as file:
#     print('opened the file!')
#     print(os.path.basename(file.name))
#     file_drive = drive.CreateFile({'title':os.path.basename(file.name) })  
#     file_drive.SetContentString(file.read()) 
#     file_drive.Upload()

file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()

for file1 in file_list:
    if file1['title'] == 'batch_5_data':
        folder_id = file1['id']
    # if file1['title'] == 'test.jpg':
    #     file1.GetContentFile(os.path.join('/media/ac12/Data/tagging_tool' ,file1['title']))
print(folder_id)
file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(folder_id)}).GetList()

for file1 in file_list:
    if file1['title'] == 'files':
        images_folder_id = file1['id']
    if file1['title'] == 'labels':
        labels_folder_id = file1['id']
print(images_folder_id)
print(labels_folder_id)

images_file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(images_folder_id)}).GetList()
labels_file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(labels_folder_id)}).GetList()
print(len(images_file_list))
print(len(labels_file_list))
# img_ind = 0
# for image in images_file_list:
#     print(img_ind, image['title'])
#     img_ind += 1
#     image.GetContentFile(os.path.join('/media/ac12/Data/tagging_tool/batch_5/images' ,image['title']))
#     for c in image.http.connections.values():
#         c.close()
# print('Downloaded {} images\n'.format(img_ind))
# label_ind = 0
# for label in labels_file_list:
#     print(label_ind, label['title'])
#     label_ind += 1
#     label.GetContentFile(os.path.join('/media/ac12/Data/tagging_tool/batch_5/labels' ,label['title']))
#     for c in label.http.connections.values():
#         c.close()
# print('Downloaded {} labels\n'.format(label_ind))

# video0.mkv_094411_color.png