from minio import Minio
from PIL import Image
from io import BytesIO
import os

class S3ImagesInvalidExtension(Exception):
    pass

class S3ImagesUploadFailed(Exception):
    pass

class S3Images(object):
    
    """Usage:
    
        s3image = S3Images(s3_endpoint='minio:9000', 
                          access_key='minioadmin', 
                          secret_key='minioadmin',
                          secure=False)

        image = s3image.from_s3('bucket', 'pythonlogo.png')
        
        image.to_s3(image, 'bucket', 'pythonlogo2.png')
    """
    
    def __init__(self, s3_endpoint, access_key, secret_key, secure):
        self.s3 = Minio(s3_endpoint,
                        access_key=access_key,
                        secret_key=secret_key,
                        secure=secure)

    def from_s3(self, bucket_name, object_name):
        file_byte_string = self.s3.get_object(bucket_name, object_name).data
        return Image.open(BytesIO(file_byte_string))
    

    def to_s3(self, img, bucket, key):
        buffer = BytesIO()
        img.save(buffer, self.__get_safe_ext(key))
        buffer.seek(0)
        sent_data = self.s3.put_object(Bucket=bucket, Key=key, Body=buffer)
        if sent_data['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise S3ImagesUploadFailed('Failed to upload image {} to bucket {}'.format(key, bucket))
        
    def __get_safe_ext(self, key):
        ext = os.path.splitext(key)[-1].strip('.').upper()
        if ext in ['JPG', 'JPEG']:
            return 'JPEG' 
        elif ext in ['PNG']:
            return 'PNG' 
        else:
            raise S3ImagesInvalidExtension('Extension is invalid') 