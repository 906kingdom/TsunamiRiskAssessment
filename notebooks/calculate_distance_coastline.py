from global_ddm import globe

def get_distance_to_coast_ddm(latitude, longitude):
    """
    global_ddm kütüphanesini kullanarak bir noktadan en yakın kıyıya 
    (kara veya su) olan mesafeyi yaklaşık olarak hesaplar.
    
    Bu fonksiyon ~5km çözünürlüklü bir grid kullanır.

    :param latitude: Enlem (float veya numpy dizisi)
    :param longitude: Boylam (float veya numpy dizisi)
    :return: En yakın kıyıya olan mesafe (kilometre). 
             Değer > 0 ise karada, < 0 ise denizde. 
             Mutlak değerini alarak mesafeyi buluruz.
    """
    try:
        # get_ddm fonksiyonu doğrudan mesafeyi km cinsinden döndürür.
        # Tek bir nokta için (skaler) veya numpy dizileri için çalışır.
        distance_km = globe.get_ddm(latitude, longitude)
        
        # Fonksiyonun tanımı gereği:
        # distance_km > 0 ise, nokta karadadır ve değer denize olan mesafedir.
        # distance_km < 0 ise, nokta denizdedir ve değer karaya olan mesafedir.
        # Biz her zaman "en yakın kıyıya" olan mesafeyi istediğimiz için
        # mutlak değerini alırız.
        return abs(distance_km)
    
    except Exception as e:
        print(f"global_ddm hatası: {e}")
        return None