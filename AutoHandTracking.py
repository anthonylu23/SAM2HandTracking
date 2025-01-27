import cv2
import mediapipe as mp
import numpy as np
from sam2.build_sam import build_sam2_video_predictor
import supervision as sv

mp_hands = mp.solutions.hands #Initializing mediapipe hands
hands = mp_hands.Hands()

colors = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700'] #Setup for mask annotator
mask_annotator = sv.MaskAnnotator(
    color=sv.ColorPalette.from_hex(colors),
    color_lookup=sv.ColorLookup.TRACK)

def hand_detection_function(frame_rgb, width, height):
    image = frame_rgb

    if image is not None:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Convert to RGB

        results = hands.process(image_rgb) #Process image

        hand_landmarks_list = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    if id == 0: #Check for wrists (This way we can minimize the amount of landmarks for efficiency will making sure we can track seperate hands)
                        hand_landmarks_list.append((lm.x * width, lm.y * height)) #Store landmark coordinates 

        hand_landmarks_array = np.array(hand_landmarks_list, dtype=np.float32) 
        return hand_landmarks_array

def sam2_initialize(input_path):
    checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt" #Initialize checkpoint/config for SAM2
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

    sam_predictor = build_sam2_video_predictor(model_cfg, checkpoint) 
    inference_state = sam_predictor.init_state(video_path=input_path) #Initialize inference state

    return sam_predictor, inference_state

def vid_initialize(input_path, output_path):
    cap = cv2.VideoCapture(input_path) #Initialize video capture
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*"X264") 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height)) #Initialize video out

    return cap, num_frames, width, height, fps, out

def sam2_reset(sam_predictor, inference_state):
    sam_predictor.reset_state(inference_state) #Reset inference state

def vid_reset(cap, out):
    cap.release()
    out.release()

def sam2_prompt_gather(input_path, cap, width, height, single_hand = False):
    num_lmks = 0
    while num_lmks <= 1: #We keep checking frames until we find one with multiple hands. Otherwise, there is a possibility of only one hand being tracked
        ret, frame = cap.read() 
        if not ret:
            single_hand = True 
            cap.release()  
            cap = cv2.VideoCapture(input_path)
            ret, frame = cap.read() 
        frame_rgb =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = hand_detection_function(frame_rgb, width, height) #Use function from part 1 to get list of landmarks (wrist points) from each frame

        point_coords = landmarks
        num_lmks = len(point_coords)
        if single_hand:
            break

    prompts = [] 
    for id in range(num_lmks): #Storing the landmarks as prompts and respective labels
        point = np.array([point_coords[id]], dtype=np.int32)
        labels = np.array([1], dtype=np.int32)
        prompts.append((point, labels))
    return prompts

def sam2_prompting(input_path, cap, width, height, sam_predictor, inference_state):
    frame_idx = 0

    prompts = sam2_prompt_gather(input_path, cap, width, height)
    
    obj_id = 0
    masks = [0] * len(prompts)
    for prompt in prompts: #New prompt given to SAM2 for each hand
        _, _, masks[obj_id] = sam_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx = frame_idx,
            obj_id = obj_id,
            points = prompt[0],
            labels = prompt[1]
        )
        obj_id += 1

def sam2_propagation(cap, out, sam_predictor, inference_state):
    for out_frame_idx, out_obj_ids, mask_logits in sam_predictor.propagate_in_video(inference_state): #Prompt propogation
        ret, frame = cap.read() #Open the next frame
        if not ret:
            break
        masks = (mask_logits > 0.0).cpu().numpy()
        N, X, H, W = masks.shape #Using supervision to annotate masks onto each frame
        masks = masks.reshape(N * X, H, W)
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks,
            tracker_id=np.array(out_obj_ids)
        )
        frame = mask_annotator.annotate(frame, detections)
        out.write(frame)

def hands_in_video_segmentation(input_path, output_path):
    sam_predictor, inference_state = sam2_initialize(input_path)
    
    cap, num_frames, width, height, fps, out = vid_initialize(input_path, output_path)

    sam2_prompting(input_path, cap, width, height, sam_predictor, inference_state)

    cap.release()  # close and re-open for propagation
    cap = cv2.VideoCapture(input_path)

    sam2_propagation(cap, out, sam_predictor, inference_state)

    sam2_reset(sam_predictor, inference_state)
    vid_reset(cap, out)
     
    print(f"Processed video saved to {output_path}")

