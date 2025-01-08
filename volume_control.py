import cv2
import mediapipe as mp
import math
import osascript

# MediaPipe el algılama modelini başlat
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Kamerayı başlat
print("Kamera başlatılıyor...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

def set_volume(volume_percentage):
    volume = float(volume_percentage)
    osascript.osascript(f"set volume output volume {volume}")

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

print("Program başlatıldı. Çıkmak için 'q' tuşuna basın.")

while True:
    success, img = cap.read()
    if not success:
        print("Kamera görüntüsü alınamadı!")
        break
    
    # Görüntüyü ayna gibi yansıt
    img = cv2.flip(img, 1)
    
    # Görüntüyü BGR'den RGB'ye çevir
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # El tespiti yap
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # El işaretlerini çiz
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Başparmak ve işaret parmağı konumlarını al
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            
            # Koordinatları piksel değerlerine çevir
            h, w, c = img.shape
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            
            # İki parmak arasındaki mesafeyi hesapla
            distance = calculate_distance(thumb_x, thumb_y, index_x, index_y)
            
            # Mesafeyi ses seviyesine dönüştür (0-100 arası)
            # Mesafe arttıkça ses artar, azaldıkça azalır
            volume = int((distance - 20) / 180 * 100)
            volume = max(0, min(100, volume))  # Ses seviyesini 0-100 arasında tut
            
            # Ses seviyesini ayarla
            set_volume(volume)
            
            # Mesafe ve ses seviyesini ekranda göster
            cv2.putText(img, f'Volume: {volume}%', (10, 70), 
                       cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            
            # Parmak uçlarını görselleştir
            cv2.circle(img, (thumb_x, thumb_y), 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (index_x, index_y), 15, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 3)

    # Görüntüyü göster
    cv2.imshow("Volume Control", img)
    
    # 'q' tuşuna basılırsa programı sonlandır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Program kapatılıyor...")
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()