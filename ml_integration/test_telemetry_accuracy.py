import time
import json
import urllib.request
import urllib.error
from realtime_adapter import RealTimeFeatureExtractor

def test_feature_extractor_accuracy():
    print("--- 1. Backend Özellik Çıkarımı (Feature Extractor) Doğruluk Testi ---")
    
    # Sentetik veri: 5 saniye içinde 10 fare hareketi, 2 tıklama, 5 tuş basımı
    base_time = int(time.time() * 1000)
    
    mouse_events = []
    # 10 hareket (x ve y 10'ar piksel artacak)
    for i in range(10):
        mouse_events.append({"x": 100 + i*10, "y": 100 + i*10, "type": "move", "time": base_time + i*500})
    
    # 2 tıklama
    mouse_events.append({"x": 200, "y": 200, "type": "click", "time": base_time + 1000})
    mouse_events.append({"x": 200, "y": 200, "type": "click", "time": base_time + 2000})
    
    keyboard_events = []
    # 'a' tuşuna 5 kere basıp çekme
    for i in range(5):
        keyboard_events.append({"key": "a", "type": "press", "time": base_time + i*1000})
        keyboard_events.append({"key": "a", "type": "release", "time": base_time + i*1000 + 100})
        
    window_events = [{"app_name": "chrome", "time": base_time}]
    
    payload = {
        "mouse_events": mouse_events,
        "keyboard_events": keyboard_events,
        "window_events": window_events
    }
    
    extractor = RealTimeFeatureExtractor()
    features = extractor.process_payload(payload)
    
    # Beklentiler ile kıyaslama
    print(f"Beklenen Mouse Hareket Sayısı: 10, Bulunan: {features.get('mouse_move_count')}")
    print(f"Beklenen Mouse Tıklama Sayısı: 2, Bulunan: {features.get('mouse_click_count')}")
    print(f"Beklenen Klavye Basım Sayısı: 5, Bulunan: {features.get('keydown_count')}")
    
    if (features.get('mouse_click_count') == 2 and 
        features.get('mouse_move_count') == 10 and 
        features.get('keydown_count') == 5):
        print("✅ Özellik Çıkarımı Doğru Çalışıyor!\n")
    else:
        print("❌ Özellik Çıkarımı Hatalı!\n")

def test_api_endpoint_accuracy():
    print("--- 2. Uçtan Uca API Tampon Bellek (Buffer) Testi ---")
    
    url = "http://localhost:8000/api/telemetry"
    # Her testte temiz bir oturum (session) başlatmak için benzersiz bir kullanıcı ID'si kullanıyoruz
    test_user_id = f"test_user_{int(time.time())}"
    
    payload = {
        "user_id": test_user_id,
        "mouse_events": [{"x": 100, "y": 100, "type": "click", "time": int(time.time() * 1000)}],
        "keyboard_events": [],
        "window_events": []
    }
    
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
    
    try:
        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode('utf-8'))
        print("API Yanıtı:", data)
        # Eğer buffer'da mouse olayı varsa veya buffer boşalmış ama bir tahmin döndürülmüşse başarılı say
        if data.get("buffer_sizes", {}).get("mouse", 0) >= 1 or data.get("prediction") is not None:
            print("✅ API Telemetriyi Başarıyla Alıyor ve Tamponluyor / İşliyor!\n")
        else:
            print("❌ API Telemetri Alma Sırasında Hata Var!\n")
    except urllib.error.URLError:
        print("❌ Backend (main.py) açık değil. Lütfen 'python main.py' komutuyla başlatın.\n")

if __name__ == "__main__":
    test_feature_extractor_accuracy()
    test_api_endpoint_accuracy()
