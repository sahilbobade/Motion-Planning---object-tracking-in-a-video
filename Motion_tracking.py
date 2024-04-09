import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

#for smoothing the graphs
def smooth(data, window_size=10):
    if len(data) >= window_size:
        data_array = np.array(data)
        weights = np.ones(window_size) / window_size
        return np.convolve(data_array, weights, mode='valid').tolist()
    return data

#LOS calculations
def calculate_los_metrics(center1, center2, prev_center1, prev_center2, dt):
    dx = center2[0] - center1[0]
    dy = center2[1] - center1[1]
    distance = np.sqrt(dx ** 2 + dy ** 2)

    angle = -np.degrees(np.arctan2(dy, dx))

    if prev_center1 and prev_center2 and dt > 0:
        prev_dx = prev_center2[0] - prev_center1[0]
        prev_dy = prev_center2[1] - prev_center1[1]
        prev_distance = np.sqrt(prev_dx ** 2 + prev_dy ** 2)
        range_rate = (distance - prev_distance) / dt

        prev_angle = -np.degrees(np.arctan2(prev_dy, prev_dx))
        angular_rate = (angle - prev_angle) / dt
    else:
        range_rate = None
        angular_rate = None

    return distance, angle, range_rate, angular_rate

#for storing information
trajectories1, velocities1, accelerations1 = [], [], []
trajectories2, velocities2, accelerations2 = [], [], []
smoothed_distances, smoothed_angles, smoothed_range_rates, smoothed_angular_rates = [], [], [], []

time_array = [0]

#main tracking and analysis loop
def track_and_analyze(video_path, output_folder):
    global trajectories1, velocities1, accelerations1, trajectories2, velocities2, accelerations2, time_array
    global smoothed_distances, smoothed_angles, smoothed_range_rates, smoothed_angular_rates 
    cap = cv2.VideoCapture(video_path)
    tracker1 = cv2.TrackerMIL_create()
    tracker2 = cv2.TrackerMIL_create()
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        cap.release()
        return

    bbox1 = cv2.selectROI("Select First Object to Track", frame, fromCenter=False, showCrosshair=True)
    bbox2 = cv2.selectROI("Select Second Object to Track", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows() 
    tracker1.init(frame, bbox1)
    tracker2.init(frame, bbox2)

    distances, angles, range_rates, angular_rates = [], [], [], []
    prev_time = None
    prev_center1 = prev_center2 = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        dt = current_time - prev_time if prev_time is not None else 0
        time_array.append(time_array[-1] + dt)
        

        success1, bbox1 = tracker1.update(frame)
        success2, bbox2 = tracker2.update(frame)

        if success1 and success2:
            center1 = (int(bbox1[0] + bbox1[2] / 2), int(bbox1[1] + bbox1[3] / 2))
            center2 = (int(bbox2[0] + bbox2[2] / 2), int(bbox2[1] + bbox2[3] / 2))
            trajectories1.append(center1)
            trajectories2.append(center2)

            # Drawing trajectories
            for point in trajectories1:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)
            for point in trajectories2:
                cv2.circle(frame, point, 2, (255, 0, 0), -1)

            distance, angle, range_rate, angular_rate = calculate_los_metrics(center1, center2, prev_center1,
                                                                              prev_center2, dt)
            distances.append(distance)
            angles.append(angle)
            if range_rate is not None: range_rates.append(range_rate)
            if angular_rate is not None: angular_rates.append(angular_rate)


            if len(trajectories1) > 1 and dt > 0:
                dx1, dy1 = center1[0] - trajectories1[-2][0], center1[1] - trajectories1[-2][1]
                velocity1 = np.sqrt(dx1 ** 2 + dy1 ** 2) / dt
                if velocity1 != 0:
                    velocities1.append(velocity1)
                if len(velocities1) > 1:
                    acceleration1 = (velocities1[-1] - velocities1[-2]) / dt
                    accelerations1.append(acceleration1)

            if len(trajectories2) > 1 and dt > 0:
                dx2, dy2 = center2[0] - trajectories2[-2][0], center2[1] - trajectories2[-2][1]
                velocity2 = np.sqrt(dx2 ** 2 + dy2 ** 2) / dt
                if velocity2 != 0:
                    velocities2.append(velocity2)
                if len(velocities2) > 1:
                    acceleration2 = (velocities2[-1] - velocities2[-2]) / dt
                    accelerations2.append(acceleration2)

         
            smoothed_distances = smooth(distances, window_size=10)
            smoothed_angles = smooth(angles, window_size=10)
            smoothed_range_rates = smooth(range_rates, window_size=70) if range_rates else []
            smoothed_angular_rates = smooth(angular_rates, window_size=70) if angular_rates else []

            cv2.putText(frame, f"LOS Dist: {smoothed_distances[-1]:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
            cv2.putText(frame, f"LOS Angle: {smoothed_angles[-1]:.2f} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
            if smoothed_range_rates:
                cv2.putText(frame, f"LOS Range Rate: {smoothed_range_rates[-1]:.2f}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if smoothed_angular_rates:
                cv2.putText(frame, f"LOS Angular Rate: {smoothed_angular_rates[-1]:.2f} deg/s", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(frame,
                        f"V1: {velocities1[-1]:.2f}, A1: {accelerations1[-1]:.2f}" if velocities1 and accelerations1 else "V1: N/A, A1: N/A",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame,
                        f"V2: {velocities2[-1]:.2f}, A2: {accelerations2[-1]:.2f}" if velocities2 and accelerations2 else "V2: N/A, A2: N/A",
                        (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            prev_center1, prev_center2 = center1, center2

        prev_time = current_time
        

        cv2.imshow("Object Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


track_and_analyze("final.mp4",
                  "frames/")  

#smoothing for plotting
velocities1 = smooth(velocities1, window_size= 70)
velocities2 = smooth(velocities2, window_size= 60)
time_array1 = time_array[0:len(velocities1)]
time_array2 = time_array[0:len(velocities2)]


accelerations1 = smooth(accelerations1, window_size= 40)
accelerations2 = smooth(accelerations2,window_size= 40)
accelerations1 = smooth(accelerations1, window_size= 40)
accelerations2 = smooth(accelerations2,window_size= 40)
time_array_acc1 = time_array[0:len(accelerations1)]
time_array_acc2 = time_array[0:len(accelerations2)]

#plots
plt.figure(1)
plt.plot(time_array1, velocities1, label='Velocity1')
plt.plot(time_array2, velocities2, label='Velocity2')
plt.title('Velocities Vs Time')
plt.xlabel('Time')
plt.ylabel('Velocities')
plt.legend()

plt.figure(2)
plt.plot(   time_array_acc1, accelerations1, label='Acceleration1')
plt.plot(   time_array_acc2, accelerations2, label='Acceleration2')
plt.title('Acceleration vs Time')
plt.xlabel('Acceleration')
plt.ylabel('Time')
plt.legend()

time_dist = time_array[0:len(smoothed_distances)]
plt.figure(3)
plt.plot(   time_dist, smoothed_distances, label='LOS Distance')
plt.title('LOS distance vs Time')
plt.xlabel('X-axis')
plt.ylabel('Time')
plt.legend()

time_ang = time_array[0: len(smoothed_angles)]
plt.figure(4)
plt.plot(   time_ang, smoothed_angles, label='LOS Angle')
plt.title('LOS Angle vs Time')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

time_rate = time_array[0:len(smoothed_range_rates)]
plt.figure(5)
plt.plot(   time_rate, smoothed_range_rates, label='LOS Rate')
plt.title('LOS Rate vs Time')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

time_ang_rate =time_array[0:len(smoothed_angular_rates)]
plt.figure(6)
plt.plot(   time_ang_rate, smoothed_angular_rates, label='LOS Angle Rate')
plt.title('LOS Angle Rate vs Time')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()    

plt.show()



