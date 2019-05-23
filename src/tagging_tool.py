from tkinter import *
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import cv2
import os, glob, string
from shutil import copyfile
import numpy as np
import pickle
from math import sin, cos, atan2, sqrt, pi

# radius of the earth
R = 6.371*(10**6)

class sequence:
    def __init__(self, canvas=None, step=None):
        self.alive = False
        self.start_ind = -1
        self.end_ind = -1
        self.canvas=canvas
        self.step = step
     
    def activate(self, start_ind, num_imgs):
        self.alive = True
        self.start_ind = start_ind
        self.end_ind = start_ind
        x1, y1 = 40 + start_ind / (num_imgs - 1) * 520, 415
        self.strat_point = self.canvas.create_oval(x1-3, y1-3, x1+3, y1+3, fill='blue')
        self.line = self.canvas.create_line(x1, y1, x1, y1, fill='blue')
        self.end_point = self.canvas.create_oval(x1-3, y1-3, x1+3, y1+3, fill='blue')
    
    def deactivate(self, end_ind):
        self.alive = FALSE
        self.end_ind = end_ind
        # x1, y1 = 40 + self.start_ind / (num_imgs - 1) * 520, 415
        # x2, y2 = 40 + end_ind / (num_imgs - 1) * 520, 415
        # self.end_point = self.canvas.create_oval(x2-3, y2-3, x2+3, y2+3, fill='blue')
        # self.line = self.canvas.create_line(x1, y1, x2, y2, fill='blue')

    def expand_sequence(self, end_ind, num_imgs):
        self.canvas.delete(self.line)
        self.canvas.delete(self.end_point)
        x1, y1 = 40 + self.start_ind / (num_imgs - 1) * 520, 415
        x2, y2 = 40 + end_ind / (num_imgs - 1) * 520, 415
        self.end_point = self.canvas.create_oval(x2-3, y2-3, x2+3, y2+3, fill='blue')
        self.line = self.canvas.create_line(x1, y1, x2, y2, fill='blue')

    def is_alive(self):
        return self.alive

    def sequence_str(self):
        return 'Image {} to {}'.format(self.start_ind*self.step, self.end_ind*self.step)

    def destroy_plot(self):
        self.canvas.delete(self.strat_point)
        self.canvas.delete(self.line)
        self.canvas.delete(self.end_point)

    def re_plot(self, num_imgs, start_ind, end_ind):
        self.start_ind, self.end_ind = start_ind, end_ind
        x1, y1 = 40 + self.start_ind / (num_imgs - 1) * 520, 415
        x2, y2 = 40 + self.end_ind / (num_imgs - 1) * 520, 415
        self.strat_point = self.canvas.create_oval(x1-3, y1-3, x1+3, y1+3, fill='blue')
        self.end_point = self.canvas.create_oval(x2-3, y2-3, x2+3, y2+3, fill='blue')
        self.line = self.canvas.create_line(x1, y1, x2, y2, fill='blue')

    def to_pickle(self):
        return (self.start_ind, self.end_ind)


class tagging_tool(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        master.geometry("1320x530")
        self.sequences = []
        self.init_window()
        master.bind('s',lambda event: self.previous_image())
        master.bind('w',lambda event: self.next_image())

    
############################################
#       initializing the main window       #
############################################
    def init_window(self):
        self.master.title("tagging_tool")
        self.pack(fill=BOTH, expand=1)

        self.quit = Button(self, text="QUIT", fg="red",height=40, width=100, command=self.quit_func)
        self.quit.place(x=1210, y=470, height=40, width=100)

        self.pre_load_videos_button = Button(self, text="Pre-load Videos", height=40, width = 250, command=self.pre_load_videos)
        self.pre_load_videos_button.place(x=780, y=470, height=40, width=250)

        # self.test = Button(self, text="test", command=self.test_button)
        # self.test.pack(side="bottom")
        # self.reset()
        self.init_sub_frames()
        self.img_iter = -1



############################################
#                   reset                  #
############################################
    def reset(self):
        self.video_file = ''
        self.video_name = ''
        self.img_folder = ''
        self.img_files = []
        self.clear_select_listbox()
        self.video_handle = None
        self.img_iter = -1
        self.num_imgs = 0
        for s in self.sequences:
            s.destroy_plot()
        self.sequences = []
        self.cur_sequence = sequence()
        

############################################
#          initializing sub-frames         #
############################################
    def init_sub_frames(self):
        self.init_operation_frame()
        self.init_image_frame()
        self.init_road_type_frame()
        self.init_road_condition_frame()
        self.init_weather_frame()
        self.init_image_quality_frame()
        self.init_scrollbar_frame()
        self.init_step_frame()
        self.init_generate_extract_frame()
        
    # initializing operation frame
    def init_operation_frame(self):
        # button size, position
        w, h, x, y = 120, 25, 10, 5

        # initialize frame
        self.operation_frame = Frame(self,width=150, height=465, bd=2, relief=GROOVE)
        self.operation_frame.place(x=5, y=5, width=150, height=465)
        op_f = self.operation_frame

        self.operation_label = Label(op_f, text='Operations')
        self.operation_label.place(x=x, y=y, width=w, height=h)
        y += 30

        # initialize buttons
        self.load_video_button = Button(op_f, text='Load Video', command=self.load_video_and_auto_tag)
        self.load_video_button.place(x=x, y=y, width=w, height=h)
        y += 30
        self.load_folder_button = Button(op_f, text='Load folder', command=self.load_folder)
        self.load_folder_button.place(x=x, y=y, width=w, height=h)
        y += 60

        self.previous_image_button = Button(op_f, text='Previous image', command=self.previous_image)
        self.previous_image_button.place(x=x, y=y, width=w, height=h)
        y += 30
        self.next_image_button = Button(op_f, text='Next image', command=self.next_image)
        self.next_image_button.place(x=x, y=y, width=w, height=h)
        y += 60

        self.save_tags_button = Button(op_f, text='Save tags', command=self.save_tags)
        self.save_tags_button.place(x=x, y=y, width=w, height=h)
        y += 30
        self.clear_tags_button = Button(op_f, text='Clear tags', command=self.clear_tags)
        self.clear_tags_button.place(x=x, y=y, width=w, height=h)
        y += 60

        self.start_sequence_button = Button(op_f, text='Start sequence', command=self.start_sequence)
        self.start_sequence_button.place(x=x, y=y, width=w, height=h)
        y += 30
        self.end_sequence_button = Button(op_f, text='End sequence', command=self.end_sequence)
        self.end_sequence_button.place(x=x, y=y, width=w, height=h)
        y += 30
        self.clear_sequence_button = Button(op_f, text='Clear sequence', command=self.clear_sequence)
        self.clear_sequence_button.place(x=x, y=y, width=w, height=h)
        y += 60

        self.reset_all_button = Button(op_f, text='Reset all', command=self.reset_all)
        self.reset_all_button.place(x=x, y=y, width=w, height=h)

    # initializing image frame
    def init_image_frame(self):
        ############################
        # load = cv2.resize(cv2.cvtColor(cv2.imread('m.jpg'), cv2.COLOR_BGR2RGB), (600, 400))
        # render = ImageTk.PhotoImage(image = Image.fromarray(load))
        # h, w, _ = load.shape
        ############################
        self.image_frame = Frame(self, width=610, height=465, bd=2, relief=GROOVE)
        self.image_frame.place(x=160, y=5, width=610, height = 465)

        self.image_frame_strvar = StringVar()
        self.image_frame_strvar.set('')

        # self.image_frame_label = Label(self.image_frame, text='Image')
        # self.image_frame_label.place(x=245, y=5, width=120, height=20)

        self.image_iter_strvar = StringVar()
        self.image_iter_strvar.set('0')

        self.image_iter_entry_label = Label(self.image_frame, text='Image index: ')
        self.image_iter_entry_label.place(x=170, y=5)
        self.image_iter_entry = Entry(self.image_frame, textvariable=self.image_iter_strvar, justify=CENTER, state='readonly')
        self.image_iter_entry.place(x=280, y=5)

        self.image_canvas = Canvas(self.image_frame, width=600, height=430)
        self.image_canvas.place(x=5, y=30, width=600, height=430)   

    # initializing road type selection frame
    def init_road_type_frame(self):
        self.road_type_frame = Frame(self, width=135, height=395, bd=2, relief=GROOVE)
        self.road_type_frame.place(x=770, y=5, width=135, height=395)
        
        self.road_type_label = Label(self.road_type_frame, text='Road Type')
        self.road_type_label.place(x=5, y=5, height=25, width=120)
        
        self.road_type_list = Listbox(self.road_type_frame,activestyle='none', selectmode=MULTIPLE, height=11, selectborderwidth=8, bd=0, exportselection=False)
        rt_l = self.road_type_list
        for s in self.road_type_strings:
            rt_l.insert(END, s)
        rt_l.place(x=5, y=35, height=350, width=120)
        rt_l.bind('<<ListboxSelect>>', lambda x : self.select_road_type())
            
    def select_road_type(self):
        self.road_type_selected = np.zeros(11, dtype=int)
        self.road_type_selected[list(self.road_type_list.curselection())]=1

    # initializing road condition selection frame
    def init_road_condition_frame(self):
        self.road_condition_frame = Frame(self, width=135, height=395, bd=2, relief=GROOVE)
        self.road_condition_frame.place(x=905, y=5, width=135, height=395)

        self.road_condition_label = Label(self.road_condition_frame, text='Road Condition')
        self.road_condition_label.place(x=5, y=5, height=25, width=120)

        self.road_condition_list = Listbox(self.road_condition_frame,activestyle='none', selectmode=MULTIPLE, height=11, selectborderwidth=8, bd=0, exportselection=False)
        rc_l = self.road_condition_list
        for s in self.road_condition_strings:
            rc_l.insert(END, s)
        rc_l.place(x=5, y=35, height=350, width=120)
        rc_l.bind('<<ListboxSelect>>', lambda x : self.select_road_condition())

    def select_road_condition(self):
        self.road_condition_selected = np.zeros(6, dtype=int)
        self.road_condition_selected[list(self.road_condition_list.curselection())]=1

    # initializing weather selection frame
    def init_weather_frame(self): 
        self.weather_frame = Frame(self, width=135, height=395, bd=2, relief=GROOVE)
        self.weather_frame.place(x=1040, y=5, width=135, height=395)

        self.weather_label = Label(self.weather_frame, text='Weather')
        self.weather_label.place(x=5, y=5, height=25, width=120)

        self.weather_list = Listbox(self.weather_frame,activestyle='none', selectmode=MULTIPLE, height=11, selectborderwidth=8, bd=0, exportselection=False)
        w_l = self.weather_list
        for s in self.weather_strings:
            w_l.insert(END, s)
        w_l.place(x=5, y=35, height=350, width=120)
        w_l.bind('<<ListboxSelect>>', lambda x : self.select_weather())

    def select_weather(self):
        self.weather_selected = np.zeros(5, dtype=int)
        self.weather_selected[list(self.weather_list.curselection())]=1

    # initializing image quality selectino frame
    def init_image_quality_frame(self):
        self.image_quality_frame = Frame(self, width=135, height=395, bd=2, relief=GROOVE)
        self.image_quality_frame.place(x=1175, y=5, width=135, height=395)

        self.image_quality_label = Label(self.image_quality_frame, text='Image Quality')
        self.image_quality_label.place(x=5, y=5, height=25, width=120)

        self.image_quality_list = Listbox(self.image_quality_frame,activestyle='none', selectmode=MULTIPLE, height=11, selectborderwidth=8, bd=0, exportselection=False)
        iq_l = self.image_quality_list
        for s in self.image_quality_strings:
            iq_l.insert(END, s)
        iq_l.place(x=5, y=35, height=350, width=120)
        iq_l.bind('<<ListboxSelect>>', lambda x : self.select_image_quality())

    def select_image_quality(self):
        self.image_quality_selected = np.zeros(4, dtype=int)
        self.image_quality_selected[list(self.image_quality_list.curselection())]=1

    # initializing scrollbar frame
    def init_scrollbar_frame(self):
        self.scrollbar_frame = Frame(self, width=610, height=40, bd=2, relief=GROOVE)
        self.scrollbar_frame.place(x=160, y=470, width=610, height = 40)

        self.scrollbar_canvas = Canvas(self.scrollbar_frame, width=600, height=40, scrollregion=(0,0,600,0))
        self.scrollbar_canvas.place(x=5, y=5, width=600, height=30)
        

        self.scrollbar = Scrollbar(self.scrollbar_frame, orient=HORIZONTAL, width=300, command=self.scrollbar_func)
        self.scrollbar.place(x=5, y=5, height=20, width=600)
        self.scrollbar_canvas.config(xscrollcommand=self.scrollbar.set)
        self.scrollbar_canvas.xview_moveto(0)

    # initializing step frame
    def init_step_frame(self):
        self.step_frame = Frame(self, width=150, height=40, bd=2, relief=GROOVE)
        self.step_frame.place(x=5, y=470, width=150, height=40)

        self.step_strvar = StringVar()
        self.step = 30
        self.step_strvar.set(str(self.step))
        
        self.step_label = Label(self.step_frame, text='Step')
        self.step_label.pack()

        self.step_entry = Entry(self.step_frame, textvariable=self.step_strvar, justify=CENTER, state='readonly')
        self.step_entry.pack()

    def init_generate_extract_frame(self):
        self.generate_extract_frame = Frame(self, width = 540, height=65, bd=2, relief=GROOVE)
        self.generate_extract_frame.place(x=770, y=405, height=65, width=540)

        self.generate_list_button = Button(self.generate_extract_frame, text='Generate label list', command=self.generate_list)
        self.generate_list_button.place(x=10, y=5, height=45, width=250)   

        self.extract_buttun = Button(self.generate_extract_frame, text='Extract to folder', command=self.extract_buttun_func)
        self.extract_buttun.place(x=280, y=5, width=250, height=45)    
    
############################################
#             buttons functions            #
############################################

    def load_video(self):
        value =  filedialog.askopenfilename(initialdir = '/mnt/storage/public/mcity_data/Batch5',title = "Please select video",filetypes = (("MKV files","*.mkv"),("AVI files","*.avi"),("MP4 files","*.mp4*"), ("All files","*.*")))
        if value is () or value is '':
            return
        self.reset()
        self.video_file = value
        self.video_handle = cv2.VideoCapture(self.video_file)
        self.video_dir, self.video_name = os.path.split(self.video_file)
        self.video_name = os.path.splitext(self.video_name)[0]
        self.img_folder = self.video_dir + '/' + self.video_name + '_frames'
        self.num_imgs = int(self.video_handle.get(cv2.CAP_PROP_FRAME_COUNT))
        self.sequence_file_name = self.img_folder + '/sequences_info.pickle'
        
        # extract frames
        if not os.path.isdir(self.img_folder):
            step = simpledialog.askinteger('Input', 'Step? Click \'cancel\' to use default value 30')
            if step is not None:
                self.step = step
                self.step_strvar.set(str(self.step))
            os.mkdir(self.img_folder)
            message = self.show_message('Loading video')
            self.num_img_digit_count, temp = 0, self.num_imgs
            while temp > 0:
                temp = temp // 10
                self.num_img_digit_count += 1
            ind = 0
            success, cur_img = self.video_handle.read()
            while success:
                if ind % self.step == 0:
                    cv2.imwrite(self.img_folder + '/' + self.video_name + '_frame_' + str(ind).zfill(self.num_img_digit_count) + '.jpg', cur_img)
                ind += 1
                success, cur_img = self.video_handle.read()
            self.close_message(message)
            self.img_files = sorted(glob.glob(self.img_folder + '/*.jpg'))
            self.num_imgs = len(self.img_files)
        else:
            self.img_files = sorted(glob.glob(self.img_folder + '/*.jpg'))
            self.num_imgs = len(self.img_files)
            loc1, loc2 = self.img_files[0].find('frame_'), self.img_files[0].find('.jpg')
            loc3, loc4 = self.img_files[1].find('frame_'), self.img_files[1].find('.jpg')
            self.step = int(self.img_files[1][loc3+6:loc4]) - int(self.img_files[0][loc1+6:loc2])
            self.step_strvar.set(str(self.step))
            self.show_message('Video already loaded! Step is {}.'.format(self.step))

        self.img_iter += 1           
        self.show_img()
        self.try_load_tags()
        self.scrollbar_canvas.config(scrollregion=(0,0,100*self.num_imgs,0))
        self.scrollbar.update()
        x, y = self.scrollbar.get()
        self.scrollbar_len = y-x
        self.cur_sequence = sequence(self.image_canvas, self.step)
        self.video_handle.release()

    def load_video_and_auto_tag(self):
        value =  filedialog.askopenfilename(initialdir = '/mnt/storage/public/mcity_data/Batch5',title = "Please select video",filetypes = (("MKV files","*.mkv"),("AVI files","*.avi"),("MP4 files","*.mp4*"), ("All files","*.*")))
        if value is () or value is '':
            return
        self.reset()
        self.video_file = value
        self.video_handle = cv2.VideoCapture(self.video_file)
        self.video_dir, self.video_name = os.path.split(self.video_file)
        self.video_name = os.path.splitext(self.video_name)[0]
        self.img_folder = self.video_dir + '/' + self.video_name + '_frames'
        self.num_imgs = int(self.video_handle.get(cv2.CAP_PROP_FRAME_COUNT))
        self.sequence_file_name = self.img_folder + '/sequences_info.pickle'

        self.load_gps_get_frame_ind()
        
        if not os.path.isdir(self.img_folder):
            os.mkdir(self.img_folder)
            message = self.show_message('Loading video')
            self.num_img_digit_count, temp = 0, self.num_imgs
            while temp > 0:
                temp = temp // 10
                self.num_img_digit_count += 1
            for ind in self.image_frame_ind:
                self.video_handle.set(1, ind)
                success, cur_img = self.video_handle.read()
                cv2.imwrite(self.img_folder + '/' + self.video_name + '_frame_' + str(ind).zfill(self.num_img_digit_count) + '.jpg', cur_img)
                # print(self.img_folder + '/' + self.video_name + '_frame_' + str(ind).zfill(self.num_img_digit_count) + '.jpg')
            self.close_message(message)
            self.img_files = sorted(glob.glob(self.img_folder + '/*.jpg'))
            self.num_imgs = len(self.img_files)
        else:
            self.img_files = sorted(glob.glob(self.img_folder + '/*.jpg'))
            self.num_imgs = len(self.img_files)
            self.show_message('Video already loaded! Step is {}.'.format(self.step))

        self.auto_tag()

        self.img_iter += 1           
        self.show_img()
        self.try_load_tags()
        self.scrollbar_canvas.config(scrollregion=(0,0,100*self.num_imgs,0))
        self.scrollbar.update()
        x, y = self.scrollbar.get()
        self.scrollbar_len = y-x
        self.cur_sequence = sequence(self.image_canvas, self.step)
        self.video_handle.release()
        
        
    def load_folder(self):
        value = filedialog.askdirectory(initialdir = '/mnt/storage/public/mcity_data/Batch5', title = "Please select folder")
        if value is () or value is '':
            return
        self.reset()
        self.video_dir = os.path.split(value)[0]
        self.img_folder = value
        temp = os.path.split(value)[1]
        pos = temp.find('_frames')
        self.video_name = temp[0:pos]

        self.load_gps_get_frame_ind()

        self.img_files = sorted(glob.glob(self.img_folder + '/*.jpg'))
        self.num_imgs = len(self.img_files)
        # loc1, loc2 = self.img_files[0].find('frame_'), self.img_files[0].find('.jpg')
        # loc3, loc4 = self.img_files[1].find('frame_'), self.img_files[1].find('.jpg')
        # self.step = int(self.img_files[1][loc3+6:loc4]) - int(self.img_files[0][loc1+6:loc2])
        # self.step_strvar.set(str(self.step))
        self.step = None
        self.show_message('Folder loaded! Step is {}.'.format(self.step))

        self.auto_tag()

        self.img_iter += 1
        self.show_img()
        self.try_load_tags()
        self.scrollbar_canvas.config(scrollregion=(0,0,100*self.num_imgs,0))
        self.scrollbar.update()
        x, y = self.scrollbar.get()
        self.scrollbar_len = y-x

        self.cur_sequence = sequence(self.image_canvas, self.step)
        self.sequence_file_name = self.img_folder + '/sequences_info.pickle'
        if os.path.exists(self.sequence_file_name):
            pickle_in = open(self.sequence_file_name, 'rb')
            temp = pickle.load(pickle_in)
            for info in temp:
                start_ind, end_ind = info
                self.sequences.append(sequence(self.image_canvas, self.step))
                self.sequences[-1].re_plot(self.num_imgs, start_ind, end_ind)

    def save_tags(self, img_iter=None):
        if img_iter is None:
            img_iter = self.img_iter
        if img_iter == -1:
            return
        cur_img = self.img_files[img_iter]
        output_file_name = os.path.splitext(cur_img)[0] + '.txt'
        if os.path.exists(output_file_name):
            os.remove(output_file_name)
        tags = np.concatenate((self.road_type_selected,self.road_condition_selected,self.weather_selected,self.image_quality_selected))
        if np.sum(tags) != 0:
            output_file = open(output_file_name, 'w')
            for i in range(len(tags)):
                cur_row = '{:>14}:{:<1}\n'.format(self.headings[i], tags[i])
                output_file.write(cur_row)
            output_file.close()

    def clear_tags(self, ind=None, called_by_pre_img_buttun=False):
        if ind is None:
            ind = self.img_iter
        if ind == -1:
            return
        if self.cur_sequence.is_alive() and not called_by_pre_img_buttun:
            messagebox.showerror('Error', 'Current sequence is not ended!')
            return 
        cur_img = self.img_files[ind]
        output_file_name = os.path.splitext(cur_img)[0] + '.txt'
        if os.path.exists(output_file_name):
            os.remove(output_file_name)
        if not called_by_pre_img_buttun:
            self.clear_select_listbox()

    def next_image(self):
        if self.img_iter == -1 or self.img_iter == self.num_imgs - 1:
            return
        self.save_tags()
        self.img_iter += 1
        # print(self.cur_sequence.end_ind)
        if not self.cur_sequence.is_alive():
            self.clear_select_listbox()
            self.try_load_tags()
        else:
            self.cur_sequence.end_ind = self.img_iter
            self.cur_sequence.expand_sequence(self.img_iter, self.num_imgs)
       
        howMany = self.img_iter / (self.num_imgs - 1) * (1-self.scrollbar_len)
        self.scrollbar_canvas.xview_moveto(howMany)
        self.scrollbar.update()
        self.show_img()
            
    def previous_image(self):
        if self.img_iter == -1 or self.img_iter == 0:
            return
        if not self.cur_sequence.is_alive():
            self.save_tags()
            self.clear_select_listbox()
            self.img_iter += -1
            self.try_load_tags()
        else:
            self.img_iter += -1
            if self.img_iter < self.cur_sequence.start_ind:
                messagebox.showerror('Error', 'Reached current sequence start image!')
                self.img_iter += 1
                return
            self.cur_sequence.end_ind = self.img_iter
            self.cur_sequence.expand_sequence(self.img_iter, self.num_imgs)
            
            self.clear_tags(ind=self.img_iter+1, called_by_pre_img_buttun=True)
        
        howMany = self.img_iter / (self.num_imgs - 1) * (1-self.scrollbar_len)
        self.scrollbar_canvas.xview_moveto(howMany)
        self.scrollbar.update()
        self.show_img() 

    def start_sequence(self):
        self.master.update()
        print('here')
        if self.img_iter == -1:
            return
        tags = np.concatenate((self.road_type_selected,self.road_condition_selected,self.weather_selected,self.image_quality_selected))
        print(np.sum(tags))
        if np.sum(tags) == 0:
            messagebox.showerror('Error', 'No tags selected!')
            return 
        if self.cur_sequence.is_alive():
            messagebox.showerror('Error', 'Already started a sequence!')
            return 
        self.cur_sequence.activate(self.img_iter, self.num_imgs)
        # x1, y1 = 40 + self.img_iter / (self.num_imgs - 1) * 520, 415
        # self.image_canvas.create_oval(x1-3, y1-3, x1+3, y1+3, fill='blue')

    def end_sequence(self):
        if self.img_iter == -1:
            return
        if not self.cur_sequence.is_alive():
            messagebox.showerror('Error', 'No current sequence!')
            return
        self.cur_sequence.deactivate(self.img_iter)
        self.sequences.append(self.cur_sequence)
        for iter in range(self.cur_sequence.start_ind, self.cur_sequence.end_ind):
            self.save_tags(img_iter=iter)
        pickle_out = open(self.sequence_file_name, 'wb')
        temp = []
        for s in self.sequences:
            temp.append(s.to_pickle())
        pickle.dump(temp, pickle_out)
        pickle_out.close()
        # print(len(self.sequences))
        # x1, y1 = 40 + self.cur_sequence.start_ind / (self.num_imgs - 1) * 520, 415
        # x2, y2 = 40 + self.img_iter / (self.num_imgs - 1) * 520, 415
        # self.image_canvas.create_oval(x2-3, y2-3, x2+3, y2+3, fill='blue')
        # self.image_canvas.create_line(x1, y1, x2, y2, fill='blue')
        self.cur_sequence = sequence(self.image_canvas, self.step)
        self.save_tags()
        self.next_image()

    def clear_sequence(self):
        if self.img_iter == -1:
            return
        if self.cur_sequence.is_alive():
            messagebox.showerror('Error', 'Current sequence is not ended!')
            return 
        if len(self.sequences) == 0:
            messagebox.showerror('Error', 'No sequences available!')
            return
        # toplevel
        x, y = self.master.winfo_x(), self.master.winfo_y()
        self.sequence_list_toplevel = Toplevel()
        height = len(self.sequences) * 30 + 50
        self.sequence_list_toplevel.geometry('200x%d+%d+%d' % (height ,x+560, y+250))
        self.sequence_list_toplevel.title('Please select sequence')
        # listbox
        self.sequence_list = Listbox(self.sequence_list_toplevel, activestyle='none', selectmode=MULTIPLE, height=len(self.sequences), selectborderwidth=8, bd=0, exportselection=False)
        for i in range(len(self.sequences)):
            self.sequence_list.insert(i, self.sequences[i].sequence_str())
        self.sequence_list.place(x=10, y=10, height=height-50,  width=180)
        self.sequence_list.bind('<<ListboxSelect>>', lambda x : self.select_sequence())
        # buttons
        self.delete_buttun = Button(self.sequence_list_toplevel, text='Delete', command=self.delete_squence)
        self.delete_buttun.place(x=25, y=height-35, height=30, width = 50)

        self.cancel_buttun = Button(self.sequence_list_toplevel, text='Cancel', command=self.sequence_list_toplevel.destroy)
        self.cancel_buttun.place(x=125, y=height-35, height=30, width = 50)
        # self.master.update()

    def reset_all(self):
        self.clear_all_tags()
        self.clear_all_sequence()
        self.img_iter = 0
        howMany = self.img_iter / (self.num_imgs - 1) * (1-self.scrollbar_len)
        self.scrollbar_canvas.xview_moveto(howMany)
        self.scrollbar.update()
        self.show_img()
    
    def scrollbar_func(self, *kargs):
        op, howMany = kargs[0], kargs[1]
        if op == 'scroll':
            if int(howMany) > 0:
                self.next_image()
            else: 
                self.previous_image()
        elif op == 'moveto':
            if self.cur_sequence.is_alive():
                messagebox.showerror('Error', 'Current sequence is not ended!')
                return
            x, y = self.scrollbar.get()
            self.scrollbar_canvas.xview_moveto(howMany)
            x, y = self.scrollbar.get()
            bar_len, mid = y - x, (x + y) / 2
            pos = (mid - bar_len/2)/ (1 - bar_len)
            if 1-pos < 0.5 / (self.num_imgs -1):
                pos = 1 
            self.img_iter = int( pos * (self.num_imgs -1) )
            self.show_img()
            self.clear_select_listbox()
            self.try_load_tags()

    def generate_list(self):
        outputfile = open(self.video_dir + '/'+ self.video_name+'.txt', 'w')
        outputfile.write('filename ')
        for h in self.headings:
            outputfile.write(h+' ')
        outputfile.write('\n')
        for ind in range(self.num_imgs):
            cur_img = self.img_files[ind]
            cur_file_name = os.path.splitext(cur_img)[0] + '.txt'
            if os.path.exists(cur_file_name):
                cur_file = open(cur_file_name, 'r')
                tags = []
                for line in cur_file:
                    tags.append(line[-2])
                outputfile.write(os.path.split(cur_img)[1]+' ')
                for t in tags:
                    outputfile.write(t+' ')
                outputfile.write('\n')
        self.show_message('List generated!')
        outputfile.close()

    def extract_buttun_func(self):
        output_dir = filedialog.askdirectory(initialdir = '/mnt/storage/public/mcity_data/Batch5', title = "Please select folder")
        if output_dir is () or output_dir is '':
            return
        dirs = []
        for h in self.headings:
            dirs.append(os.path.join(output_dir, h))
        dirs.append(os.path.join(output_dir, 'all_labelled'))
        for d in dirs:
            if not os.path.isdir(d):
                os.mkdir(d)
        self.show_message('Extracted to folder!')
        for ind in range(self.num_imgs):
            cur_img = self.img_files[ind]
            cur_file_name = os.path.splitext(cur_img)[0] + '.txt'
            if os.path.exists(cur_file_name):
                cur_file = open(cur_file_name, 'r')
                tags = []
                for line in cur_file:
                    tags.append(int(line[-2]))
                copied = os.path.join(dirs[-1], os.path.split(cur_img)[1])
                copyfile(cur_img, copied)
                for i in range(len(tags)):
                    if tags[i] > 0:
                        copied = os.path.join(dirs[i], os.path.split(cur_img)[1])
                        copyfile(cur_img, copied)
                
    def quit_func(self):
        if self.img_iter < 0:
            self.save_tags()
        self.master.destroy()
    
    def pre_load_videos(self):
        data_folder = filedialog.askdirectory(initialdir = 'mnt/storage/public/mcity_data/Batch5', title = "Please select folder")
        if data_folder is () or data_folder is '':
            return
        step = 30
        temp = simpledialog.askinteger('Input', 'Step? Click \'cancel\' to use default value 30')
        if temp is not None:
            step = temp
        message = self.show_message('Loading video')
        for dirpath, dirnames, filenames in os.walk(data_folder):       
            if not dirnames:
                videos_folder = dirpath
                video_files = sorted(glob.glob(videos_folder + '/*.avi'))
                for i in range(len(video_files)):
                    video_name = video_files[i]
                    img_folder = os.path.splitext(video_name)[0] + '_frames/'
                    video_name_str = os.path.splitext(os.path.split(video_name)[1])[0]
                    if not os.path.isdir(img_folder):
                        os.mkdir(img_folder)
                        video_handle = cv2.VideoCapture(video_name)
                        num_imgs = int(video_handle.get(cv2.CAP_PROP_FRAME_COUNT))
                        num_img_digit_count, temp = 0, num_imgs
                        while temp > 0:
                            temp = temp // 10
                            num_img_digit_count += 1
                        success, cur_img = video_handle.read()
                        ind = 0
                        while success:
                            if ind % step == 0:
                                cv2.imwrite(img_folder + '/' + video_name_str + '_frame_' + str(ind).zfill(num_img_digit_count) + '.jpg', cur_img)
                            ind += 1
                            success, cur_img = video_handle.read()
        self.close_message(message)
############################################
#              other utilities             #
############################################
    def show_img(self):
        cur_img = self.img_files[self.img_iter]
        load = cv2.resize(cv2.cvtColor(cv2.imread(cur_img), cv2.COLOR_BGR2RGB), (600, 400))
        render = ImageTk.PhotoImage(image = Image.fromarray(load))
        self.image_canvas.create_image(0, 0, image=render, anchor=NW)
        self.image_canvas.photo = render
        # self.image_iter_strvar.set(str(self.img_iter * self.step))
        self.image_iter_strvar.set(str(self.image_frame_ind[self.img_iter]))
        self.image_canvas.update()
                
    def show_message(self, content):
        x, y = self.master.winfo_x(), self.master.winfo_y()
        message_toplevel = Toplevel()
        message_toplevel.geometry('200x100+%d+%d' % (x+560, y+250))
        message_toplevel.title('Message')
        message = Message(message_toplevel, text=content, width=100, justify=CENTER)
        message.pack(expand=1)
        self.master.update()
        return message_toplevel

    def close_message(self, message_toplevel):
        message_toplevel.destroy()

    def clear_selected(self):
        self.road_type_selected = np.zeros(11, dtype=int)
        self.road_condition_selected = np.zeros(6, dtype=int)
        self.weather_selected = np.zeros(5, dtype=int)
        self.image_quality_selected = np.zeros(4, dtype=int)

    def clear_select_listbox(self):
        self.clear_selected()
        self.road_type_list.select_clear(0, END)
        self.road_condition_list.select_clear(0, END)
        self.weather_list.select_clear(0, END)
        self.image_quality_list.select_clear(0, END)

    def try_load_tags(self):
        cur_img = self.img_files[self.img_iter]
        cur_file_name = os.path.splitext(cur_img)[0] + '.txt'
        if os.path.exists(cur_file_name):
            cur_file = open(cur_file_name, 'r')
            tags = []
            for line in cur_file:
                tags.append(int(line[-2]))
            self.road_type_selected = np.array(tags[0:11])
            self.road_condition_selected = np.array(tags[11:17])
            self.weather_selected = np.array(tags[17:22])
            self.image_quality_selected = np.array(tags[22:26])
            for t in range(0, 11):
                if tags[t] > 0:
                    self.road_type_list.selection_set(t)
                # self.road_type_list.update()
            for t in range(11, 17):
                if tags[t] > 0:
                    self.road_condition_list.selection_set(t-11)
            for t in range(17, 22):
                if tags[t] > 0:
                    self.weather_list.selection_set(t-17)
            for t in range(22, 26):
                if tags[t] > 0:
                    self.image_quality_list.selection_set(t-22)
            cur_file.close()

    def select_sequence(self):
            self.temp_selected_sequences = list(self.sequence_list.curselection())
    
    def delete_squence(self):
        length = len(self.sequences)
        remain_sequence= [self.sequences[i] for i in range(length) if i not in self.temp_selected_sequences]
        delete_sequences = [self.sequences[i] for i in range(length) if i in self.temp_selected_sequences]
        for s in delete_sequences:
            s.destroy_plot()
            for ind in range(s.start_ind, s.end_ind+1):
                self.clear_tags(ind)
        self.sequences = remain_sequence
        pickle_out = open(self.sequence_file_name, 'wb')
        temp = []
        for s in self.sequences:
            temp.append(s.to_pickle())
        pickle.dump(temp, pickle_out)
        pickle_out.close()
        self.temp_selected_sequences = []
        self.sequence_list_toplevel.destroy()

    def clear_all_tags(self):
        for i in range(self.num_imgs):
            self.clear_tags(i)

    def clear_all_sequence(self):
        if self.cur_sequence.is_alive():
            self.end_sequence()
        for s in self.sequences:
            s.destroy_plot()
        self.sequences = []

    def get_distance(self, pt1, pt2):
        pt1 = np.array(pt1) / 180 * pi
        pt2 = np.array(pt2) / 180 * pi
        d_phi = pt2[0] - pt1[0]
        d_lambda = pt2[1] - pt1[1]
        a = (np.sin(d_phi/2))**2 + cos(pt1[0])*cos(pt2[0])*((sin(d_lambda/2))**2)
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    def load_gps_get_frame_ind(self):
        gps_data = pickle.load(open(os.path.join(self.video_dir,'vehicle_gps.txt'), 'rb'))
        timestemp_file = open(os.path.join(self.video_dir,self.video_name+'.txt'), 'rb')
        self.video_timestamp = []
        for line in timestemp_file:
            l = line.replace(b',', b' ').split()
            self.video_timestamp.append(int(l[1])*10**(-3))
        self.gps_data = []
        self.image_frame_ind = []
        ind, i = 0, 0
        max_diff = 0
        bad = 0
        while i < len(gps_data):
            cur_gps_time = gps_data[i][3] +  gps_data[i][4]*10**(-9)
            while ind < len(self.video_timestamp) and self.video_timestamp[ind] < cur_gps_time:
                ind += 1
            if ind == len(self.video_timestamp):
                break
            if self.video_timestamp[ind] - cur_gps_time < 0.5:
                diff = self.video_timestamp[ind] - cur_gps_time
                self.image_frame_ind.append(ind)
                self.gps_data.append(gps_data[i])
            i += 1
        
    def auto_tag(self):
        self.adj_dis = []
        for i in range(len(self.gps_data)-1):
            dis = self.get_distance(self.gps_data[i][0:2],self.gps_data[i+1][0:2])
            self.adj_dis.append(dis)
            
        for i in range(len(self.gps_data)-10):
            dis = self.get_distance(self.gps_data[i][0:2],self.gps_data[i+10][0:2])
            # print(self.image_frame_ind[i], dis, np.sum(self.adj_dis[i:i+10]))
            diff = np.sum(self.adj_dis[i:i+10]) - dis
            # if diff > 1:
            #     print(self.image_frame_ind[i+5], 'curve')
            # else:
            #     print(self.image_frame_ind[i+5], 'straight')
            # print(self.image_frame_ind[i+5], self.gps_data[i+10][2] - self.gps_data[i][2])


############################################
#                  variables               #
############################################
    road_type_strings = 'Straight Curve Circle Split Merge Branch-in Branch-out Uphill Downhill Intersection Other'.split()
    road_condition_strings = 'Wet Snow Cracked Patched Shadow Occlusion'.split()
    weather_strings = 'Rain Snow Fog Sunny Normal'.split()
    image_quality_strings = 'Blur Dark Dizzy Exposure'.split()
    headings = ('1_Straight 1_Curve 1_Cirvle 1_Split 1_Merge 1_Branch-in 1_Branch-out 1_Uphill 1_Downhill 1_Intersection 1_Other ' \
        + '2_Wet 2_Snow 2_Cracked 2_Patched 2_Shadow 2_Occlusion ' \
        + '3_Rain 3_Snow 3_Fog 3_Sunny 3_Normal ' \
        + '4_Blur 4_Dark 4_Dizzy 4_Exposure').split()  
    
##################################################################
##################################################################


# def main():
#     root = Tk()
#     app = tagging_tool(master=root)
#     root.mainloop()
# 
# 
# if __name__ == "__main__":
# 	main()
