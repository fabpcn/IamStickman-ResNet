import numpy as np
import cv2
import sys
import time
import os

#from train.metrics import max_coords

def draw_vt(vt, shape, color):
	thickness = max(int(np.round(5*shape[0]/224)), 1)
	output = np.zeros(shape)
	for i in range(int(len(vt)/2)):
		cv2.circle(output,(int(np.round(vt[2*i])), int(np.round(vt[2*i +1]))), thickness, color, -1)
	return output

def create_visualization(pred, image, vt):
	W,H,C = image.shape
	output = np.zeros((W*3, H , C))
	if len(pred.shape) > 1:
		tmp = []
		for elem in pred:
			x,y = max_coords(elem)
			tmp.append(x)
			tmp.append(y)
		pred = tmp
	output[:W,:,:] = 255 * image
	if vt is not None:
		output[W:2*W,:,:] = draw_vt(vt=vt, shape=image.shape, color = (255, 0, 0))
	output[2*W:,:,:] = draw_vt(vt=pred, shape=image.shape, color = (0, 255, 255))
	return output

def evaluate_on_a_set(DNN, image_shape, eval_set):
	preds = []
	errors = []
	if 'Dataset' in os.listdir('..'):
		if 'Test' + str(eval_set) in os.listdir('../Dataset'):
			is_labelled = False
			if 'test' + str(eval_set) + '_labels.npy' in os.listdir('../Dataset'):
				labels = np.load('../Dataset/' + 'test' + str(eval_set) + '_labels.npy')
				is_labelled = True
			images = ['../Dataset/Test' + str(eval_set) + '/' + e for e in os.listdir('../Dataset/Test' + str(eval_set)) if '.png' in e]
			images.sort()
			for cpt, img in enumerate(images):
				print('\rtesting on set %i/4 : %i/%i'%(eval_set, cpt+1, len(images)), end='')
				img_ = cv2.imread(img, cv2.IMREAD_UNCHANGED)
				img = cv2.resize(cv2.imread(img, cv2.IMREAD_UNCHANGED), (image_shape, image_shape)) / 255
				pred = DNN.predict(np.expand_dims(img,axis = 0)) / image_shape
				if is_labelled:
					KP=labels[cpt].astype(float)
					KP_x = np.copy(KP[::2]) / img_.shape[0]
					KP_y = np.copy(KP[1::2]) / img_.shape[1]
					KP[::2] = KP_x
					KP[1::2] = KP_y
					errors.append(np.sum(np.abs(KP - pred)[0][KP > 0]))
					preds.append(pred)
					image = create_visualization(pred=pred[0] * image_shape, image=img, vt=KP * image_shape)
					cv2.imshow('',np.uint8(image))
					k = cv2.waitKey(33)
					if k==27:	# Esc key to stop
						sys.exit()
					elif k==32: # Esc space to pause
						k = cv2.waitKey(33)
						while(k != 32):
							k = cv2.waitKey(33)
							continue
					input()
				else:
					preds.append(pred)
					image = create_visualization(pred=pred[0] * image_shape, image=img, vt=None)
					cv2.imshow('',np.uint8(image))
					k = cv2.waitKey(33)
					if k==27:	# Esc key to stop
						sys.exit()
					elif k==32: # Esc space to pause
						k = cv2.waitKey(33)
						while(k != 32):
							k = cv2.waitKey(33)
							continue
			print()
		if is_labelled:
			return np.mean(errors)
		return preds

def visu_eval_DNN(DNN, image_shape):
	return (evaluate_on_a_set(DNN=DNN, image_shape=image_shape, eval_set=1),
			evaluate_on_a_set(DNN=DNN, image_shape=image_shape, eval_set=2),
			evaluate_on_a_set(DNN=DNN, image_shape=image_shape, eval_set=3),
			evaluate_on_a_set(DNN=DNN, image_shape=image_shape, eval_set=4))



'''''''''''''''''''''''''Ajout'''''''''''''''''''
def evaluate_on_val_set(DNN, image_shape):
	preds = []
	errors = []
	if 'Valset' in os.listdir('..'):
		if 'Val' in os.listdir('../Valset'):
			is_labelled = False
			if 'Val_labels.npy' in os.listdir('../Valset'):
				labels = np.load('../Valset/' + 'Val_labels.npy')
				print(labels)
				is_labelled = True
			images = ['../Valset/Val/' + e for e in os.listdir('../Valset/Val/') if '.png' in e]
			images.sort()
			for cpt, img in enumerate(images):
				print('\rtesting on val_set: %i/%i'%( cpt+1, len(images)), end='')
				img_ = cv2.imread(img, cv2.IMREAD_UNCHANGED)
				img = cv2.resize(cv2.imread(img, cv2.IMREAD_UNCHANGED), (image_shape, image_shape)) / 255
				pred = DNN.predict(np.expand_dims(img,axis = 0)) / image_shape
				if is_labelled:
					KP=labels[cpt].astype(np.float)

					KP_x = np.copy(KP[::2]) / img_.shape[0]
					KP_y = np.copy(KP[1::2]) / img_.shape[1]
					#print(KP_x, KP_y)
					KP[::2] = KP_x
					KP[1::2] = KP_y
					errors.append(np.sum(np.abs(KP - pred)[0][KP > 0]))
					preds.append(pred)
					image = create_visualization(pred=pred[0] * image_shape, image=img, vt=KP * image_shape)
					cv2.imshow('',np.uint8(image))
					k = cv2.waitKey(33)
					if k==27:	# Esc key to stop
						sys.exit()
					elif k==32: # Esc space to pause
						k = cv2.waitKey(33)
						while(k != 32):
							k = cv2.waitKey(33)
							continue
					input()
				else:
					preds.append(pred)
					image = create_visualization(pred=pred[0] * image_shape, image=img, vt=None)
					cv2.imshow('',np.uint8(image))
					k = cv2.waitKey(33)
					if k==27:	# Esc key to stop
						sys.exit()
					elif k==32: # Esc space to pause
						k = cv2.waitKey(33)
						while(k != 32):
							k = cv2.waitKey(33)
							continue

			np.save("predicts_val", preds)
			#print("Preds_shape = ", len(preds))
			print()
		print("Saved")	
		if is_labelled:
			return np.mean(errors)

		return None

def visu_eval_val_DNN(DNN, image_shape):
	return evaluate_on_val_set(DNN=DNN, image_shape=image_shape)


def visu_train_DNN(DNN, train_set, image_shape):
	num_images = train_set.__len__()
	print(num_images)
	KPs = []
	preds = []
	errors = []
	print(num_images)
	for i in range(int(num_images)):
		img,KP=train_set.__getitem__(index=i)
		KP = KP.astype(float)
		for j in range(img.shape[0]):
			#print(KP.shape)
			#KPs.append(KP[0][j])
			img_ = img[j]

			KP_x = np.copy(KP[j][::2]) / img_.shape[0]
			KP_y = np.copy(KP[j][1::2]) / img_.shape[1]
			#print(KP_x, KP_y)
			KP[j][::2] = KP_x
			KP[j][1::2] = KP_y
			pred = DNN.predict(np.expand_dims(img[j]/255,axis = 0)) / image_shape
			# print(pred.shape)
			# print(KP.shape)
			# print(np.sum(np.abs(KP[j] - pred)[0]).shape)
			errors.append(np.sum(np.abs(KP[j] - pred)[0][KP[j] > 0]))
			preds.append(pred)
			image = create_visualization(pred=pred[0] * image_shape, image=img[j], vt=KP[j] * image_shape)
			cv2.imshow('',np.uint8(image))
			k = cv2.waitKey(33)
			if k==27:	# Esc key to stop
				sys.exit()
			elif k==32: # Esc space to pause
				k = cv2.waitKey(33)
				while(k != 32):
					k = cv2.waitKey(33)
					continue
			input()
	return None
		
    


# import os
# import cv2
# import numpy as np

# def evaluate_on_a_set(DNN, image_shape, eval_set):
# 	preds = []
# 	errors = []
# def visualization(DNN, val_set, show_image):
# 	errors = []
# 	for step in range(val_set.__len__()):
# 		images, KP = val_set.__getitem__(index=step)
# 		preds = DNN.predict(images)
# 		for i in range(len(images)):
# 			errors.append(np.sqrt(np.sum((preds[i]-KP[i])**2)))
# 			if show_image:
# 				image = create_visualization(pred=preds[i], image=images[i], vt=KP[i])
# 				cv2.imshow('',np.uint8(image))
# 				k = cv2.waitKey(33)
# 				if k==27:	# Esc key to stop
# 					sys.exit()
# 				elif k==32: # Esc space to pause
# 					print('\rvisualisation : %i/%i'%(step, val_set.__len__()), end='') 
# 					k = cv2.waitKey(33)
# 					while(k != 32):
# 						k = cv2.waitKey(33)
# 						continue
# 			else:
# 				print('results :')
# 				print('\t err', errors[-1])
# 				print('\t',np.round(preds[i]).astype(int))
# 				print('\t',KP[i])
# 				# time.sleep(1)
# 	return errors[-1]


if __name__ == '__main__':
    import sys
    sys.path.append('c:\\Users\\fabic\\Desktop\\Cours Td Cloud\\M2\\Signaux_Sociaux\\IamStickman-ResNet\\IamStickman-main')
    from generator.stickman import stick_man_generator
    
    train = stick_man_generator(
		batch_size = 64, 
		set_of_data = 'train', 
		p_circles= 0.5, 
		p_squares=0.3, 
		p_real= 0.9, 
		input_shape = (224, 224, 3))
    visu_train_DNN(None, train, 224)