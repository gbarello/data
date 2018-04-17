import numpy as np
#import imageio
import glob
import pickle
import tensorflow as tf
import os
import sys

def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
def import_data(data_type):
    if data_type == "HumanActionVids":
        return import_HAV_data(data_type)
    if data_type == "two_moving_shapes":
        return make_TMS_data(data_type)
    if data_type == "N_moving_shapes":
        return make_NMS_data(data_type)
    else:
        print("data type not recognized")
        exit()

def save_metadata(direc,data):
      F = open(direc + "metadata.pkl","wb")
      pickle.dump(data,F)
      F.close()

def load_metadata(direc):
      F = open(direc + "metadata.pkl","rb")
      out = pickle.load(F)
      F.close()
      return out

def import_HAV_data(data_type,v_len = 20): 
      files = glob.glob("/home/gbarello/data/HumanActionVids/videos/*")
      
      people = ["0" + str(k) for k in range(1,10)] + ["1" + str(k) for k in range(10)] + ["2" + str(k) for k in range(6)]
      activities = ["boxing","walking","jogging","running","handclapping","handwaving"]
      cond = ["d1","d2","d3","d4"]
      
      def get_vid(p,a,c):
            filename = '/home/gbarello/data/HumanActionVids/videos/person'+p+'_'+a+'_'+c+'_uncomp.avi'
            
            try:
                  vid = imageio.get_reader(filename,  'ffmpeg')
                  
                  out = []
                  
                  for im in vid.iter_data():
                        out.append(im.mean(axis = 2))
                        
                  return np.array(out)
            except:
                  return np.array([])
          
          
          
      train_p = people[:15]
      test_p = people[15:20]
      val_p = people[20:]

      def get_lab(L):
            return activities.index(L[1])
#            return L[0] + "_" + L[1] + "_" + L[2]

      print("get training data")
      train = [get_vid(p,a,c) for p in train_p for a in activities for c in cond]
      trlab = [get_lab([p,a,c]) for p in train_p for a in activities for c in cond]
      train_2 = np.array([train[k][j-v_len:j] for k in range(len(train)) for j in range(v_len,len(train[k]),v_len)])
      trlab = np.array([trlab[k] for k in range(len(train)) for j in range(v_len,len(train[k]),v_len)],np.float32)
      train = np.float32(train_2)/255

      print("get test data")
      test = [get_vid(p,a,c) for p in test_p for a in activities for c in cond]
      telab = [get_lab([p,a,c]) for p in test_p for a in activities for c in cond]
      test_2 = np.array([test[k][j-v_len:j] for k in range(len(test)) for j in range(v_len,len(test[k]),v_len)])
      telab = np.array([telab[k] for k in range(len(test)) for j in range(v_len,len(test[k]),v_len)],np.float32)
      test = np.float32(test_2)/255
      
      print("get validation data")
      val = [get_vid(p,a,c) for p in val_p for a in activities for c in cond]
      valab = [get_lab([p,a,c]) for p in val_p for a in activities for c in cond]
      val_2 = np.array([val[k][j-v_len:j] for k in range(len(val)) for j in range(v_len,len(val[k]),v_len)])
      valab = np.array([valab[k] for k in range(len(val)) for j in range(v_len,len(val[k]),v_len)],np.float32)
      val = np.float32(val_2)/255
      
      return [train,trlab],[test,telab],[val,valab]

def make_TMS_data(data_type,v_len = 100): 
      from MovingShapes import moving_shape_generators as gen

      shapes = ["square","triangle"]
      fsize = 30

      print("get training data")
      train = [gen.make_random_video(2,fsize,v_len) for k in range(1000)]
      trdat = np.array([k[0] for k in train],np.float32)
      trlabel = np.array([[np.append(i,np.append(k[2],k[3])) for i in k[1]] for k in train],np.float32)

      print("get test data")
      test = [gen.make_random_video(2,fsize,v_len) for k in range(100)]
      tedat = np.array([k[0] for k in test],np.float32)
      telabel = np.array([[np.append(i,np.append(k[2],k[3])) for i in k[1]] for k in test],np.float32)

      print("get val data")
      val = [gen.make_random_video(2,fsize,v_len) for k in range(100)]
      vadat = np.array([k[0] for k in val],np.float32)
      valabel = np.array([[np.append(i,np.append(k[2],k[3])) for i in k[1]] for k in val],np.float32)
      
      return [trdat,trlabel],[tedat,telabel],[vadat,valabel]

def make_NMS_data(data_type,v_len = 100,nmax = 3,fsize = 30): 
      from MovingShapes import moving_shape_generators as gen

      shapes = ["square","triangle"]

      print("get training data")
      
      train = [gen.make_random_video(nmax,fsize,v_len) for k in range(1000)]      
      trdat = np.array([k[0] for k in train],np.float32)
      trlabel = np.array([[np.append(np.reshape(i,[-1]),np.append(k[2],k[3])) for i in k[1]] for k in train],np.float32)

      print("get test data")
      test = [gen.make_random_video(nmax,fsize,v_len) for k in range(100)]
      tedat = np.array([k[0] for k in test],np.float32)
      telabel = np.array([[np.append(np.reshape(i,[-1]),np.append(k[2],k[3])) for i in k[1]] for k in test],np.float32)

      print("get val data")
      val = [gen.make_random_video(nmax,fsize,v_len) for k in range(100)]
      vadat = np.array([k[0] for k in val],np.float32)
      valabel = np.array([[np.append(np.reshape(i,[-1]),np.append(k[2],k[3])) for i in k[1]] for k in val],np.float32)
      
      return [trdat,trlabel],[tedat,telabel],[vadat,valabel]

def save_data(fname,train,test,val):
      dname = "/home/gbarello/data/datasets/"+fname 
      os.mkdir(dname)
    
      save_metadata(dname,{"train_size": train[0].shape,
                           "test_size": test[0].shape,
                           "val_size": val[0].shape,
                           "label_size":train[1].shape[1:]})
      
      write_to_tf_file(dname+"/train.tfrecords","train",train[0],train[1])
      write_to_tf_file(dname+"/test.tfrecords","test",test[0],test[1])
      write_to_tf_file(dname+"/val.tfrecords","val",val[0],val[1])
      
def write_to_tf_file(name,dtag,data,label):

      writer = tf.python_io.TFRecordWriter(name)
      
      for i in range(len(data)):
            # print how many images are saved every 1000 images
            if not i % 1000:
                  print('Train data: {}/{}'.format(i, len(data)))
                  sys.stdout.flush()
                  
            # Load the image
            img = data[i]
            lab = label[i]
            print(img.shape)
            # Create a feature
            feature = {dtag+'/label': _bytes_feature(tf.compat.as_bytes(lab.tostring())),
                       dtag+'/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
                                                            
      writer.close()

def open_data(datatype,mode,epochs):
      direc = "/home/gbarello/data/datasets/"+datatype

      metadata = load_metadata(direc)
      print(metadata)

      if mode == "test":
            return get_from_tf_file(direc+"/test.tfrecords","test",metadata,epochs)
      elif mode == "train":
            return get_from_tf_file(direc+"/train.tfrecords","train",metadata,epochs)
      elif mode == "val":
            return get_from_tf_file(direc+"/val.tfrecords","val",metadata,epochs)
      else:
            print("mode not recognized")
            return 0

def get_from_tf_file(name,dtag,meta,epochs):
      
      data_path = name

      feature = {dtag + '/image': tf.FixedLenFeature([], tf.string),
                 dtag + '/label': tf.FixedLenFeature([], tf.string)}
      
      # Create a list of filenames and pass it to a queue
      filename_queue = tf.train.string_input_producer([data_path],num_epochs = epochs)
      print(filename_queue)
      # Define a reader and read the next record
      
      reader = tf.TFRecordReader()
      _, serialized_example = reader.read(filename_queue)
      
      # Decode the record read by the reader
      features = tf.parse_single_example(serialized_example, features=feature)
      
      # Convert the image data from string back to the numbers
      image = tf.decode_raw(features[dtag+'/image'], tf.float32)
      
      # Cast label data into int32
      label = tf.decode_raw(features[dtag+'/label'], tf.float32)
      
      # Reshape image data into the original shape
      print(image.shape)
      image = tf.reshape(image, meta["train_size"][1:])
      label = tf.reshape(label, meta["label_size"])
            
      return image, label
      
def get_data(data_type,mode,epochs = None,expand = False):
      print("/home/gbarello/data/datasets/"+data_type)
      if os.path.exists("/home/gbarello/data/datasets/"+data_type) == False:
            tr,te,va = import_data(data_type)
            save_data(data_type,tr,te,va)
            
      img,lab = open_data(data_type,mode,epochs = epochs)        

      if expand:
            return tf.expand_dims(img,-1),lab
      else:
            return img,lab

if __name__ == "__main__":

      config = tf.ConfigProto(device_count = {'GPU': 0})

      with tf.Session(config = config) as sess:
            images, labels = get_data("HumanActionVids","val",epochs = 1)
            print(images,labels)
            
            # Initialize all global and local variables
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            
            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for batch_index in range(100):
                  img, lbl = sess.run([images, labels])

                  print(batch_index)
                  
            print(img.shape)
            print(lbl)
