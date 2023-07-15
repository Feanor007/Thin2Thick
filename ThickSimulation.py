import math
import time
import numpy as np
import scipy.ndimage as ndimage

class ReconstructedImage:
    def __init__(self, nx, ny):
        self.Nx = nx
        self.Ny = ny
        self.FieldOfView = 0.0
        self.XOrigin = 0.0
        self.YOrigin = 0.0
        self.Z = 0.0
        self.SliceThickness = 0.0
        self.Data = np.zeros((nx , ny))
        self.RescaledToHU = True

def in_range(x, start, end):
    return (x >= start and x <= end) or (x >= end and x <= start)

def generate_slice_locations(start, end, slice_spacing):
    slice_locations = []
    direction = (end - start) / math.fabs(end - start)

    if start == end:
        slice_locations.append(start)
    else:
        curr_slice = start
        while in_range(curr_slice, start, end):
            slice_locations.append(curr_slice)
            curr_slice += (direction * slice_spacing)
            curr_slice = round(curr_slice,1)
    return slice_locations

def generate_config(g, new_thickness=3.0, new_spacing=2.0):
    ReconVolume = {
    'StartPosition':0,
    'EndPosition':0, 
    'SliceThickness': 0,
    'SliceSpacing': 0,
    'Nx': 512,
    'Ny': 512,
    'XOrigin': 0,
    'YOrigin': 0,
    'RescaleSlope':0,
    'RescaleIntercept':0
    }
    img_pos = g[-1].ImagePositionPatient[2]
    num_slices = len(g)
    slice_interval = round(g[1].ImagePositionPatient[2] - g[0].ImagePositionPatient[2],1)
    ReconVolume['EndPosition'] = round(img_pos,1)
    ReconVolume['StartPosition'] = round(ReconVolume['EndPosition'] - ((num_slices-1) * slice_interval), 1)
    ReconVolume['SliceThickness'] = new_thickness 
    ReconVolume['SliceSpacing'] = new_spacing 
    ReconVolume['RescaleSlope'] = g[-1].RescaleSlope
    ReconVolume['RescaleIntercept'] = g[-1].RescaleIntercept
    print("Thickenning Configs:")
    print(ReconVolume)
    return ReconVolume
    
def TriangleWindow(slice_location, sample_location, slice_thickness):
    return max(0.0, 1.0-abs(slice_location-sample_location)/slice_thickness)

def InverseTriangleWindow(slice_location, sample_location, slice_thickness):
    return max(0.0, 1.0-0.615*abs(slice_location-sample_location)/slice_thickness)

def GaussianWindow(slice_location, sample_location, slice_thickness):
    sigma = slice_thickness / math.sqrt(2.2 * math.log(2)) 
    normalization = 1 #4 / (sigma * math.sqrt(2 * math.pi))
    return normalization*math.exp(-(slice_location - sample_location)**2 / (2 * sigma**2))

def Thickness(output_volume, image_stack, weight_function='T', is_dicom=True):

    # Get the locations of the output image stack
    output_slice_locations = generate_slice_locations(output_volume['EndPosition'], 
                                                      output_volume['StartPosition'], 
                                                      output_volume['SliceSpacing'])[::-1]
    # Check the location is correct
    print(f"First five generated slice locations: {output_slice_locations[:5]}")
    print(f"Last five generated slice locations: {output_slice_locations[-5:]}")
     
    final_image_stack = [ReconstructedImage(512, 512) for _ in range(len(output_slice_locations))]
    
    # Generate the thickened slices
    start = time.monotonic()
    print("Thicknessing to final images...")
    if weight_function == 'IU':
        print('Using Inverse Weight Up-Sampling Function')
    elif weight_function == 'G':
        print('Using Gaussian Weight Function')
    else:
        print('Using Triangle Weight Function')
    for i, v in enumerate(output_slice_locations):

        # Create the image and set metadata
        img = ReconstructedImage(output_volume['Nx'], output_volume['Ny'])

        img.Nx = 512
        img.Ny = 512

        img.Z = v
        img.SliceThickness = output_volume['SliceThickness']
        
        # For each input_image in the image_stack, calculate if it
        # will contribute to the current slice, if so, add it and
        # keep track of the weight
        
        total_weight = 0.0
        if is_dicom:
            for input_image in image_stack: 
                if weight_function == 'IU':
                    weight = InverseTriangleWindow(img.Z, input_image.ImagePositionPatient[2], img.SliceThickness)
                elif weight_function == 'G':
                    GaussianWindow(img.Z, input_image.ImagePositionPatient[2], img.SliceThickness)
                else:
                    weight = TriangleWindow(img.Z, input_image.ImagePositionPatient[2], img.SliceThickness)

                if weight == 0:
                    continue
                img.Data += weight * input_image.pixel_array
                total_weight += weight
        
        else:
            for input_image in image_stack: 
                if weight_function == 'IU':
                    weight = InverseTriangleWindow(img.Z, input_image.Z, img.SliceThickness)
                elif weight_function == 'G':
                    weight = GaussianWindow(img.Z, input_image.Z, img.SliceThickness)
                else:
                    weight = TriangleWindow(img.Z, input_image.Z, img.SliceThickness)

                if weight == 0:
                    continue
                img.Data += weight * input_image.Data
                total_weight += weight

        img.Data = img.Data/total_weight
        if is_dicom:
            img.Data = img.Data*input_image.RescaleSlope + input_image.RescaleIntercept
        img.Data = img.Data.astype(np.float32)

        # Save into the final image stack
        final_image_stack[i] = img

    print(f"Thicknessed images in {time.monotonic() - start}s")

    return final_image_stack

def average_thickness(files, thickness):
    # Initialize 3D numpy array
    depth = len(files) // thickness
    image_array = np.zeros((depth, 512, 512), dtype=np.float32)

    start = time.monotonic()
    print("Thicknessing to final images...")
    
    if thickness == 1:
        for i, dicom_file in enumerate(files):
            image_array[i, :, :] = (
                dicom_file.pixel_array * dicom_file.RescaleSlope + dicom_file.RescaleIntercept
            )
    else:
        # Initialize the stack
        stack = np.zeros((thickness, 512, 512))
        for i, dicom_file in enumerate(files):
            # Add current dicom_file to stack regardless
            stack[i % thickness, :, :] = (
                dicom_file.pixel_array * dicom_file.RescaleSlope + dicom_file.RescaleIntercept
            )

            # Calculate average when the stack is full
            if (i + 1) % thickness == 0:
                image_array[i // thickness, :, :] = np.mean(stack, axis=0)
                
    print(f"Thicknessed images in {time.monotonic() - start}s")
    return image_array


def Gaussian_average_thickness(files, thickness, sigma):
    # Initialize 3D numpy array
    depth = len(files) // thickness
    image_array = np.zeros((depth, 512, 512), dtype=np.float32)

    start = time.monotonic()
    print("Thicknessing to final images...")
    
    # Loop through DICOM files and stack pixel arrays
    stack = np.zeros((thickness, 512, 512))
    for i, dicom_file in enumerate(files):
        stack[i % thickness, :, :] = (
            dicom_file.pixel_array * dicom_file.RescaleSlope + dicom_file.RescaleIntercept
        )

        # When the correct amount of slices have been stacked, perform averaging and smoothing
        if (i + 1) % thickness == 0:
            averaged_image = np.mean(stack, axis=0)
            smoothed_image = ndimage.gaussian_filter(averaged_image, sigma=sigma)
            image_array[i // thickness, :, :] = smoothed_image
    
    print(f"Thicknessed images in {time.monotonic() - start}s")
    return image_array


def downsampling_thickness(files, thickness):
    # Initialize 3D numpy array
    depth = len(files) // thickness
    image_array = np.zeros((depth, 512, 512), dtype=np.float32)

    start = time.monotonic()
    print("Thicknessing to final images...")
    
    # Loop through DICOM files and stack pixel arrays
    stack = np.zeros((thickness, 512, 512))
    for i, dicom_file in enumerate(files):
        # Always add image to stack
        stack[i % thickness, :, :] = (
            dicom_file.pixel_array * dicom_file.RescaleSlope + dicom_file.RescaleIntercept
        )

        # When the correct amount of slices have been stacked, pick the center image
        if (i + 1) % thickness == 0:
            center_index = thickness // 2
            center_image = stack[center_index]
            image_array[i // thickness, :, :] = center_image
            stack = np.zeros((thickness, 512, 512))  # Reset stack
    
    print(f"Thicknessed images in {time.monotonic() - start}s")
    return image_array
