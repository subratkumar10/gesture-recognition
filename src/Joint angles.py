
import pandas
import csv
from numpy.linalg import norm
import numpy as np
import math
import glob
import os
def talker():
    file_list = glob.glob(r"D:\Research_Project\My_project_22\kinect_test_cases\*.csv")    ### input file
    print(file_list)
    for file_name in file_list:
        new_file_name = os.path.splitext( os.path.basename( file_name ) )[0] + '_ANGLES.csv'
        print(new_file_name)
        with open("D:/Research_Project/My_project_22/kinect_test_cases/ANGLES/" + new_file_name, mode='w') as file:  ### destination file
            result=pandas.read_csv(file_name , header=None) 
        #    LeftShoulder=np.empty([len(result),3])
    #        LSP = 0
    #        LSR = 0
    #        LEY = 0
    #        LSR = 0 
            z=1
            
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,lineterminator='\r')
            t1=0
            t2=0
            t3=0
            for i in range(0, len(result)):
        #        LeftShoulder[i]=result.iloc[i,5*3+1:5*3+4]
        #        print(LeftShoulder[i])
                p="Pose"
                ################################Left Shoulder Angles ###########################
                
                x_5 = result.iloc[i,4*3+1]
                y_5 = result.iloc[i,4*3+2]
                z_5 = result.iloc[i,4*3+3]
                
                x_6 = result.iloc[i,5*3+1]
                y_6 = result.iloc[i,5*3+2]
                z_6 = result.iloc[i,5*3+3]
                
    #            print(x_5 , y_5, z_5)
                
                
                LSE_x = x_6 - x_5 
                LSE_y = y_6 - y_5
                LSE_z = z_6 - z_5
                
        #        print(LSE_x , LSE_y, LSE_z)
             
                if LSE_z < 0:
                    LSP=np.arctan(LSE_y/LSE_z)
                if LSE_z > 0 and LSE_y < 0:
                    LSP=-(np.arctan(LSE_z/LSE_y)) + math.pi/2
                if LSE_z > 0 and LSE_y > 0:
                    LSP=-(np.arctan(LSE_z/LSE_y)) - math.pi/2
                
        #        print(LSP)
                    
                hip_right_x_17 = result.iloc[i,16*3+1]      #In kinect labelling 17 joint is HipRight and 13 is HipLeft
                hip_right_y_17 = result.iloc[i,16*3+2]
                hip_right_z_17 = result.iloc[i,16*3+3]
                
                hip_left_x_13 = result.iloc[i,12*3+1]
                hip_left_y_13 = result.iloc[i,12*3+2]
                hip_left_z_13 = result.iloc[i,12*3+3]
                
                Vec_h_x = hip_right_x_17  - hip_left_x_13
                Vec_h_y = hip_right_y_17 - hip_left_y_13
                Vec_h_z = hip_right_z_17 - hip_left_z_13
                
        #        print(x_13 , y_13, z_13)
                
                
                vec_Vh= np.array([Vec_h_x,Vec_h_y,Vec_h_z])
                vec_lse= np.array([LSE_x , LSE_y, LSE_z])
                vec_dot=np.dot(vec_Vh,vec_lse)
                norm_vec = norm(vec_Vh) * norm(vec_lse)
                LSR = np.arccos(vec_dot/norm_vec) - math.pi/2
        #        print(LSR)
                
                
                ############################left Elbow Angles #################################
                
                x_7 = result.iloc[i,6*3+1]
                y_7 = result.iloc[i,6*3+2]
                z_7 = result.iloc[i,6*3+3]
                
                LEW_x = x_7 - x_6
                LEW_y = y_7 - y_6
                LEW_z = z_7 - z_6
                    
        #        print(LEW_x ,LEW_y, LEW_z)  
                 
                N1 = np.cross(np.array([LEW_x,LEW_y,LEW_z]),vec_lse)
                
                
                shoulder_Right_x_9 = result.iloc[i,8*3+1]
                shoulder_Right_y_9 = result.iloc[i,8*3+2]
                shoulder_Right_z_9 = result.iloc[i,8*3+3]
                
        #        print(shoulder_Right_x_9 ,shoulder_Right_y_9, shoulder_Right_z_9)      
                
                vec_s_x = shoulder_Right_x_9 - x_5 
                vec_s_y = shoulder_Right_y_9 - y_5
                vec_s_z = shoulder_Right_z_9 - z_5
                
        #        print(vec_s_x ,vec_s_y, vec_s_z)  
                
                N2 = np.cross(np.array([vec_s_x ,vec_s_y, vec_s_z]),vec_lse)
                
        #        print(N1) 
        #        print(N1[0])
                
                if N1[0] < 0:    
                    LEY = - (np.arccos(np.dot(N1,N2)/(norm(N1)*norm(N2))))
                else:
                    LEY = np.arccos(np.dot(N1,N2)/(norm(N1)*norm(N2)))
                    
        #        print(LEY)    
                vec_dot_product =np.dot((-vec_lse),np.array([LEW_x,LEW_y,LEW_z]))
                vec_norm = norm(vec_lse) * norm(np.array([LEW_x,LEW_y,LEW_z]))
                LER = - math.pi + (np.arccos(vec_dot_product / vec_norm))
    #           print(LER)
                
                
            ############################left Hip Angles #################################           
                
                spineBase_HipLeft_x = hip_left_x_13 - result.iloc[i, 1]
                spineBase_HipLeft_y = hip_left_y_13 - result.iloc[i, 2]
                spineBase_HipLeft_z = hip_left_z_13 - result.iloc[i, 3]
                vec_spineBase_HipLeft = np.array([spineBase_HipLeft_x ,spineBase_HipLeft_y,spineBase_HipLeft_z])
                
                spineBase_spineMid_x = result.iloc[i,4] - result.iloc[i, 1]
                spineBase_spineMid_y = result.iloc[i,5] - result.iloc[i, 2]
                spineBase_spineMid_z = result.iloc[i,6] - result.iloc[i, 3]
                vec_spineBase_spineMid = np.array([spineBase_spineMid_x,spineBase_spineMid_y,spineBase_spineMid_z])
                
                vec_N_0_1_17 =	np.cross( vec_spineBase_HipLeft, vec_spineBase_spineMid )
                
                vec_R_LLT = np.cross( vec_spineBase_spineMid, vec_N_0_1_17 )
                
                vec_HipLeft_KneeLeft = np.array([result.iloc[i, 40], result.iloc[i,41],result.iloc[i,42]]) - np.array([hip_left_x_13,hip_left_y_13,hip_left_z_13] )
                
                LHR_mod = np.arccos( ( np.dot( vec_R_LLT, ( np.cross( vec_N_0_1_17, vec_HipLeft_KneeLeft ) ) ) )/( norm( vec_R_LLT ) * norm( ( np.cross( vec_N_0_1_17, vec_HipLeft_KneeLeft ) ) ) ) )
                
                LHR_cond = np.arccos( ( np.dot( vec_HipLeft_KneeLeft, vec_R_LLT ) )/( norm( vec_HipLeft_KneeLeft )* norm( vec_R_LLT ) ) )
                
                if LHR_cond <= math.pi/2:           
                    LHR = LHR_mod
                else:
                    LHR = - LHR_mod
                    
                LHP = -math.pi/2 + np.arccos( ( np.dot( vec_HipLeft_KneeLeft, vec_N_0_1_17 ) )/ ( norm( vec_HipLeft_KneeLeft ) * norm( vec_N_0_1_17 ) ) )
                
                
                
                
            ##############################left Knee Angles #################################       
                
                vec_KneeLeft_AnkleLeft = np.array( [result.iloc[i, 43], result.iloc[i,44], result.iloc[i, 45 ] ] ) - np.array( [result.iloc[i, 40], result.iloc[i,41], result.iloc[i, 42 ] ] )
#                vec_N_17_18_19 = np.cross( vec_KneeLeft_AnkleLeft, ( - vec_HipLeft_KneeLeft )  )
#                vec_R_LTh = np.cross( vec_N_17_18_19, ( - vec_HipLeft_KneeLeft ) )
                LKP = math.pi - np.arccos( np.dot( vec_KneeLeft_AnkleLeft, ( -vec_HipLeft_KneeLeft ) )/ ( norm( vec_KneeLeft_AnkleLeft )* norm( vec_HipLeft_KneeLeft ) ) )
                
                
            ####################################################################################   
                
            ################################ Right Shoulder Angles ###########################
            
                shoulder_right_x_9 = result.iloc[i,8*3+1]
                shoulder_right_y_9 = result.iloc[i,8*3+2]
                shoulder_right_z_9 = result.iloc[i,8*3+3]
                
                elbow_right_x_10 = result.iloc[i,9*3+1]
                elbow_right_y_10 = result.iloc[i,9*3+2]
                elbow_right_z_10 = result.iloc[i,9*3+3]
                
    #            print(elbow_right_x_10 ,elbow_right_y_10 ,elbow_right_z_10)
             
                
                RSE_x = elbow_right_x_10 - shoulder_right_x_9
                RSE_y = elbow_right_y_10 - shoulder_right_y_9
                RSE_z =  elbow_right_z_10 - shoulder_right_z_9
                
    #            print(LSE_x , LSE_y, RSE_z)
             
                if RSE_z < 0:
                    RSP = np.arctan(RSE_y/RSE_z)
                if RSE_z > 0 and RSE_y < 0:
                    RSP= math.pi/2 - (np.arctan(RSE_z/RSE_y))
                if RSE_z > 0 and RSE_y > 0:
                    RSP= -((np.arctan(RSE_z/RSE_y)) + math.pi/2)    
    #            print(RSP)
               
                hip_right_x_17 = result.iloc[i,16*3+1]       #In kinect labelling 17 joint is HipRight and 13 is HipLeft
                hip_right_y_17 = result.iloc[i,16*3+2]
                hip_right_z_17 = result.iloc[i,16*3+3]
                
                hip_left_x_13 = result.iloc[i,12*3+1]
                hip_left_y_13 = result.iloc[i,12*3+2]
                hip_left_z_13 = result.iloc[i,12*3+3]
                
                Vec_h_x = hip_left_x_13  - hip_right_x_17
                Vec_h_y = hip_left_y_13 - hip_right_y_17
                Vec_h_z = hip_left_z_13 - hip_right_z_17
                
    #            print(x_13 , y_13, z_13)
                
                
                vec_Vh= np.array([Vec_h_x,Vec_h_y,Vec_h_z])
                vec_rse= np.array([RSE_x , RSE_y, RSE_z])
               # print(vec_rse)
                vec_dot=np.dot(vec_Vh,vec_rse)
                norm_vec = norm(vec_Vh) * norm(vec_rse)
                RSR =  - np.arccos(vec_dot/norm_vec) + math.pi/2
    #            print(RSR)
                
                
                ############################Right Elbow Angles #################################
                
                wrist_right_x_11 = result.iloc[i,10*3+1]
                wrist_right_y_11 = result.iloc[i,10*3+2]
                wrist_right_z_11 = result.iloc[i,10*3+3]
                
    #            print(wrist_right_x_11 ,wrist_right_y_11, wrist_right_z_11)  
              
                REW_x = wrist_right_x_11 -elbow_right_x_10
                REW_y = wrist_right_y_11 - elbow_right_y_10
                REW_z = wrist_right_z_11 - elbow_right_z_10
                    
    #            print(LEW_x,LEW_y,LEW_z)  
             
                N1 = np.cross(vec_rse, np.array([REW_x,REW_y,REW_z]))
                
                x_5 = result.iloc[i,4*3+1]
                y_5 = result.iloc[i,4*3+2]
                z_5 = result.iloc[i,4*3+3]
                
                shoulder_Right_x_9 = result.iloc[i,8*3+1]
                shoulder_Right_y_9 = result.iloc[i,8*3+2]
                shoulder_Right_z_9 = result.iloc[i,8*3+3]
                
        #        print(shoulder_Right_x_9 ,shoulder_Right_y_9, shoulder_Right_z_9)      
                
                vec_s_x = shoulder_Right_x_9 - x_5  
                vec_s_y = shoulder_Right_y_9 - y_5   
                vec_s_z = shoulder_Right_z_9 - z_5  
                
        #        print(vec_s_x ,vec_s_y, vec_s_z)  
                
                N2 = np.cross(np.array([vec_s_x ,vec_s_y, vec_s_z]),vec_rse)
                
        #        print(N1) 
        #        print(N1[0])
                
                if N1[0] < 0:    
                    REY = - (np.arccos(np.dot(N1,N2)/(norm(N1)*norm(N2))))
                else:
                    REY =  (np.arccos(np.dot(N1,N2)/(norm(N1)*norm(N2))))
                    
    #            print(REY)    
    
                vec_dot_product =np.dot(vec_rse,np.array([REW_x,REW_y,REW_z]))
                vec_norm = norm(vec_rse) * norm(np.array([REW_x,REW_y,REW_z]))
                RER =  (np.arccos(vec_dot_product / vec_norm))
    #            print(RER)
                
                
                ############################Right Hip Angles #################################           
                
                spineBase_HipRight_x = hip_right_x_17 - result.iloc[i, 1]
                spineBase_HipRight_y = hip_right_y_17 - result.iloc[i, 2]
                spineBase_HipRight_z = hip_right_z_17 - result.iloc[i, 3]
                
                vec_spineBase_HipRight = np.array([spineBase_HipRight_x ,spineBase_HipRight_y,spineBase_HipRight_z])
                
                spineBase_spineMid_x = result.iloc[i,4] - result.iloc[i, 1]
                spineBase_spineMid_y = result.iloc[i,5] - result.iloc[i, 2]
                spineBase_spineMid_z = result.iloc[i,6] - result.iloc[i, 3]
                
                vec_spineBase_spineMid = np.array([spineBase_spineMid_x,spineBase_spineMid_y,spineBase_spineMid_z])
                
                vec_N_1_2_17 =	np.cross( vec_spineBase_spineMid, vec_spineBase_HipRight  )
                     
                vec_R_RLT = np.cross( vec_N_1_2_17, vec_spineBase_spineMid )
                
                vec_HipRight_KneeRight = np.array([result.iloc[i, 52], result.iloc[i,53],result.iloc[i,54]]) - np.array([hip_right_x_17,hip_right_y_17,hip_right_z_17] )
                
                RHR_mod = np.arccos( ( np.dot( vec_R_RLT, ( np.cross( vec_HipRight_KneeRight, vec_N_1_2_17  ) ) ) )/( norm( vec_R_RLT ) * norm( ( np.cross( vec_N_1_2_17, vec_HipRight_KneeRight ) ) ) ) )
                
                RHR_cond = np.arccos( ( np.dot( vec_HipRight_KneeRight, vec_R_RLT ) )/( norm( vec_HipRight_KneeRight )* norm( vec_R_RLT ) ) )
                
                if RHR_cond <= math.pi/2:
                    RHR = - RHR_mod
                else:
                    RHR = RHR_mod
                    
                RHP = - math.pi/2 + np.arccos( ( np.dot( vec_HipRight_KneeRight, vec_N_1_2_17 ) )/ ( norm( vec_HipRight_KneeRight ) * norm( vec_N_1_2_17 ) ) )
                
                
                
                
                ##############################Right Knee Angles #################################       
                
                vec_KneeRight_AnkleRight = np.array( [result.iloc[i, 55], result.iloc[i,56], result.iloc[i, 57 ] ] ) - np.array( [result.iloc[i, 52], result.iloc[i,53], result.iloc[i, 54 ] ] )
                
    #            vec_N_17_18_19 = np.cross( vec_KneeRight_AnkleRight, ( - vec_HipRight_KneeRight )  )
    #            
    #            vec_R_LTh = np.cross( vec_N_17_18_19, ( - vec_HipRight_KneeRight ) )
                
                RKP =  math.pi - np.arccos( np.dot( vec_KneeRight_AnkleRight, ( -vec_HipRight_KneeRight ) )/ ( norm( vec_KneeRight_AnkleRight )* norm( vec_HipRight_KneeRight ) ) )
                
                ########################################################################################    
                
                p=p+str(z)
                z=z+1
                
                if(t3>=1000):
    #                t3=0
                    t2=t2+1
                if(t2>=60):
                    t2=0
                    t1=t1+1
                s1="{:0>2d}".format(t1);
                s2="{:0>2d}".format(t2);
                s3="{:0>3d}".format(t3);
                s=s1+':'+s2+':'+s3
        #        writer.writerow([s,p,LSP,LSR,LEY,LER,t1,t2,t3])
                writer.writerow([s,p,LSP,LSR,LEY,LER,RSP,RSR,REY,RER,LHP,LHR,LKP,RHP,RHR,RKP,t3])
    #            print(LSP, LSR, LEY, LER,s,p)
                
                t3+=40
                
    #
    ##    vec=np.empty([3,1])
    #    vec_Vh= np.array([2,3,4])
    #    vec_lse= np.array([2,3,4])
    #    ve=np.dot(vec_Vh,vec_lse)
    #    print(ve)
    #  
    #    a= norm(vec_Vh)
    ##    print(a)
    #    
 


     
if __name__ == '__main__':
    talker()
