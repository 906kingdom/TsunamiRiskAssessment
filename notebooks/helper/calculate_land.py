import numpy as np
from global_land_mask import globe
from geopy.distance import great_circle

def is_ocean(latitude, longitude, min_uzaklik_km=10, test_noktasi_sayisi=8):
    """
    Belirtilen coğrafi koordinatın okyanusa en az min_uzaklik_km kadar uzak olup olmadığını
    bir dizi test noktası kullanarak yaklaşık olarak kontrol eder.
    
    Not: Bu, tam kıyı şeridine olan mesafeyi hesaplamaz, ancak etraftaki
    10 km'lik çevrenin karada olup olmadığını kontrol ederek bir fikir verir.
    """
    
    # 1. Koordinatın kendisinin okyanusta olup olmadığını kontrol et
    if globe.is_ocean(latitude, longitude):
        return True # Zaten suyun üzerinde

    # 2. Belirtilen min_uzaklik_km yarıçapında test noktaları oluştur
    
    # geopy.distance.great_circle mesafesini kullanarak yönlere doğru 10km uzaktaki noktaları hesaplayalım.
    baslangic_koordinati = (latitude, longitude)
    test_noktalari = []
    
    # Temel yönler ve ara yönlerde test noktaları oluştur
    aci_adim = 360 / test_noktasi_sayisi
    
    for i in range(test_noktasi_sayisi):
        aci = i * aci_adim
        
        # Orijinal noktadan belirtilen yönde ve mesafede yeni bir nokta hesapla
        # `geopy` kütüphanesini kullanabiliriz
        destination = great_circle(kilometers=min_uzaklik_km).destination(baslangic_koordinati, bearing=aci)
        test_noktalari.append((destination.latitude, destination.longitude))

    # 3. Tüm test noktalarının okyanusta olup olmadığını kontrol et
    for lat, lon in test_noktalari:
        if globe.is_ocean(lat, lon):
            # 10 km uzakta bir yerde su tespit edildi
            return True

    # Tüm test noktaları karadaysa, okyanustan 10 km uzakta olduğu varsayılır.
    return False

