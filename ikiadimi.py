try:
    from distutils.command.install_egg_info import safe_name
    import streamlit.components.v1 as components
    import json
    import cv2
    import os
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image
    import streamlit as st
    import numpy as np
    import base64
    import requests
    import json
    import os

except Exception as e:
    print(e)

# Pinata API anahtarları
PINATA_API_KEY = 'c8d2247f8167030040c9'
PINATA_SECRET_KEY = 'daa465cf735e5dbeff3365a76b63dd15c53ea4b2d5e04aa8a6a4a04018a45540'


PINATA_ENDPOINT = 'https://api.pinata.cloud/pinning/pinFileToIPFS'
PINATA_GATEWAY_PREFIX = 'https://gateway.pinata.cloud/ipfs/'

STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""

astyle = """
display: inline;
width: 200px;
height: 40px;
background: #F63366;
padding: 9px;
margin: 8px;
text-align: center;
vertical-align: center;
border-radius: 5px;

line-height: 25px;
text-decoration: none;
"""

st.set_page_config(
    page_title="Steganografi Bilimine Yeni Bir Boyut",
)

tabs = ["Bilgi Gizleme veya Gizlenen Bilgiyi Elde Etme"]
page = st.sidebar.radio("Sekmeler", tabs)


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def psnr(imageA, imageB):
    mse_val = mse(imageA, imageB)
    if mse_val == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_val = 20 * np.log10(max_pixel / np.sqrt(mse_val))
    return psnr_val

def compare_pixel_values(original_image, encoded_image):
    original_array = np.array(original_image)
    encoded_array = np.array(encoded_image)

    # Piksel farklarını bulma
    pixel_diff = np.sum(original_array != encoded_array)

    # Piksel farklarının yüzdesini hesaplama
    total_pixels = original_array.size
    percentage_diff = (pixel_diff / total_pixels) * 100

    return pixel_diff, percentage_diff


def compare_images(image1, image2):
    image_np1 = np.array(image1)
    image_np2 = np.array(image2)

    mse_val = mse(image_np1, image_np2)
    psnr_val = psnr(image_np1, image_np2)

    # Histogramları oluşturma
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(image_np1.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
    axes[0].set_title('Orijinal Resim Histogramı')
    axes[0].set_xlabel('Piksel Değerleri')
    axes[0].set_ylabel('Piksel Sayısı')
    axes[0].grid()

    # Fark görüntüsü oluşturma
    diff_image = cv2.absdiff(image_np1, image_np2)
    axes[1].imshow(diff_image, cmap='pink')
    axes[1].set_title('Fark Görüntüsü ()')
    axes[1].axis('off')

    axes[2].hist(image_np2.ravel(), bins=256, range=[0, 256], color='red', alpha=0.7)
    axes[2].set_title('Şifrelenmiş Resim Histogramı')
    axes[2].set_xlabel('Piksel Değerleri')
    axes[2].set_ylabel('Piksel Sayısı')
    axes[2].grid()

    plt.tight_layout()
    return mse_val, psnr_val, fig


def get_image_download_link(filename, img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = '<a href="data:file/png;base64,' + img_str + '" download=' + filename + ' style="' + astyle + '" target="_blank">Resmi indir</a>'
    return href


def get_key_download_link(filename, key):
    buffered = BytesIO()
    key.dump(buffered)
    key_str = base64.b64encode(buffered.getvalue()).decode()
    href = '<a href="data:file/pkl;base64,' + key_str + '" download=' + filename + ' style="' + astyle + '" target="_blank">Download Key</a>'
    return href


def modPix(pix, data):
    datalist = [format(ord(i), '08b') for i in data]
    lendata = len(datalist)
    imdata = iter(pix)
    for i in range(lendata):
        pix = [value for value in imdata.__next__()[:3] + imdata.__next__()[:3] + imdata.__next__()[:3]]
        for j in range(0, 8):
            if (datalist[i][j] == '0'):
                pix[j] &= ~(1 << 0)
            elif (datalist[i][j] == '1'):
                pix[j] |= (1 << 0)
        if (i == lendata - 1):
            pix[-1] |= (1 << 0)
        else:
            pix[-1] &= ~(1 << 0)
        pix = tuple(pix)
        yield pix[0:3]
        yield pix[3:6]
        yield pix[6:9]


def encode_enc(newimg, data):
    w = newimg.size[0]
    (x, y) = (0, 0)
    for pixel in modPix(newimg.getdata(), data):
        newimg.putpixel((x, y), pixel)
        if (x == w - 1):
            x = 0
            y += 1
        else:
            x += 1


def encode(filename, image, bytes):
    global c1, c2
    data = c1.text_area("Kodlanacak veriyi giriniz", max_chars=bytes)
    if (c1.button('Encode', key="1")):
        if (len(data) == 0):
            c1.error("Veri boş")
        else:
            c2.markdown('#')
            result = "İstenilen Mesaj, Örtü verisine kodlanmıştır."
            c2.success(result)
            c2.markdown('####')
            c2.markdown("#### Kodlanmış resim")
            c2.markdown('######')
            newimg = image.copy()
            encode_enc(newimg, data)

            # Resmi Pinata'ya yükle
            ipfs_hash = upload_to_pinata(newimg)
            if ipfs_hash:
                c2.info("Resim Verisi Başarı ile Hash Edildi. IPFS Hash::::::::::: " + ipfs_hash)

                filename = 'encoded_' + filename
                image_np = np.array(image)
                newimg_np = np.array(newimg)

                MSE, PSNR, histogram_fig = compare_images(image_np, newimg_np)  # MSE, PSNR ve histogramları oluşturma

                # Histogramları gösterme
                c2.pyplot(histogram_fig)

                msg = "MSE: " + str(MSE) + " | PSNR: " + str(PSNR)

                if PSNR > 40 and 0<MSE<6:  # Belirli bir PSNR eşiği belirleyerek güvenilirliği değerlendirme
                    c2.success("Yapı güvenilir.")
                else:
                    c2.error("Yapı güvenilir değil, yeni bir resim kullanın")

                c2.warning(msg)
                c2.markdown("#")
                c2.markdown(get_image_download_link(filename, newimg), unsafe_allow_html=True)
            else:
                c2.error("Resim Hash ediliriken bir hata oluştu.")
def decode_from_pinata(ipfs_hash):
    image = fetch_from_pinata(ipfs_hash)
    if image:
        data = decode(image)
        st.subheader("kodu çözülmüş metin")
        st.write(data)
        st.image(image, channels="BGR")
    else:
        st.error("Hatalı Hash Kodu Girdiniz Lütfen Doğru Kodu Girdiğinizden Emin Olun.")


def decode(image):
    data = ''
    imgdata = iter(image.getdata())
    while (True):
        pixels = [value for value in imgdata.__next__()[:3] + imgdata.__next__()[:3] + imgdata.__next__()[:3]]
        binstr = ''
        for i in pixels[:8]:
            if (i % 2 == 0):
                binstr += '0'
            else:
                binstr += '1'
        data += chr(int(binstr, 2))
        if (pixels[-1] % 2 != 0):
            return data


def upload_to_pinata(image):
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()

    files = {'file': ('image.png', img_bytes)}
    headers = {
        'pinata_api_key': PINATA_API_KEY,
        'pinata_secret_api_key': PINATA_SECRET_KEY
    }

    response = requests.post(PINATA_ENDPOINT, files=files, headers=headers)

    if response.status_code == 200:
        pinata_response = response.json()
        return pinata_response['IpfsHash']
    else:
        return None


def fetch_from_pinata(ipfs_hash):
    ipfs_url = f"{PINATA_GATEWAY_PREFIX}{ipfs_hash}"
    response = requests.get(ipfs_url)

    if response.status_code == 200:
        img_buffer = BytesIO(response.content)
        image = Image.open(img_buffer)
        return image
    else:
        return None


def main():
    global c1, c2, d1, d2

    if page == "Bilgi Gizleme veya Gizlenen Bilgiyi Elde Etme":
        if st.checkbox("Resim Gizleme ve Gizli Resmi Çözme"):
            st.sidebar.title("Dijital Görüntü İşleme Projesi")
            md = ("")
            st.sidebar.markdown(md, unsafe_allow_html=True)
            info = """
                    # Resim Steganografisi
                    Steganografi, nesnenin içinde saklı hiçbir bilgi yokmuş gibi izleyiciyi aldatacak şekilde nesnelerin içindeki 
                    bilgileri gizleme çalışması ve uygulamasıdır. 
                    Sadece hedeflenen alıcının görebilmesi için bilgileri açık bir şekilde gizler. """
            fileTypes = ["png", "jpg"]
            fileTypes1 = ["pkl"]
            choice = st.radio('Seçim', ["Encode", "Decode"])
            if (choice == "Encode"):
                c1, c2 = st.columns(2)
                file = c1.file_uploader("Kapak Resmini Yükle", type=fileTypes, key="fu1")
                show_file = c1.empty()
                if not file:
                    show_file.info("Lütfen bir dosya türü yükleyin: " + ", ".join(["png", "jpg"]))
                    return
                im = Image.open(BytesIO(file.read()))
                filename = file.name
                w, h = im.size
                bytes = (w * h) // 3
                c1.info("maksimum veri: " + str(bytes) + " Bytes")
                encode(filename, im, bytes)
                content = file.getvalue()
                if isinstance(file, BytesIO):
                    show_file.image(file)
                file.close()
            elif (choice == "Decode"):
                ipfs_hash = st.text_input("IPFS Hash'ini Girin:")
                if st.button('Decode', key="4"):
                    decode_from_pinata(ipfs_hash)


if __name__ == "__main__":
    main()