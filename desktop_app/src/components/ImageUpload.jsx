import React, { useRef } from 'react';

function ImageUpload({ targetImages, setTargetImages, selectedImageIndex, setSelectedImageIndex, serverUrl }) {
    const fileInputRef = useRef(null);

    const handleFileSelect = (e) => {
        const files = Array.from(e.target.files);
        if (files.length + targetImages.length > 10) {
            alert('Maximum 10 images allowed');
            return;
        }

        // Read files and create preview URLs
        files.forEach(file => {
            const reader = new FileReader();
            reader.onload = (event) => {
                setTargetImages(prev => [...prev, {
                    file,
                    url: event.target.result,
                    name: file.name
                }]);
            };
            reader.readAsDataURL(file);
        });
    };

    const removeImage = (index) => {
        setTargetImages(prev => prev.filter((_, i) => i !== index));
        if (selectedImageIndex >= targetImages.length - 1) {
            setSelectedImageIndex(Math.max(0, targetImages.length - 2));
        }
    };

    return (
        <div className="space-y-4">
            <div>
                <h2 className="text-lg font-semibold text-white mb-2">Target Images</h2>
                <p className="text-sm text-gray-400 mb-4">
                    Upload 1-10 post-surgery preview images
                </p>
                <p className="text-xs text-yellow-400 mb-4">
                    Use a clear frontal target photo (full face, no extreme crop).
                </p>
            </div>

            {/* Upload Button */}
            <button
                onClick={() => fileInputRef.current?.click()}
                disabled={targetImages.length >= 10 || !serverUrl}
                className="w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
            >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                Upload Images ({targetImages.length}/10)
            </button>

            <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                onChange={handleFileSelect}
                className="hidden"
            />

            {!serverUrl && (
                <p className="text-xs text-yellow-500">⚠️ Set server URL in settings first</p>
            )}

            {/* Image Grid */}
            <div className="space-y-2">
                {targetImages.map((img, index) => (
                    <div
                        key={index}
                        onClick={() => setSelectedImageIndex(index)}
                        className={`relative group cursor-pointer rounded-lg overflow-hidden border-2 transition-all ${selectedImageIndex === index
                                ? 'border-blue-500 ring-2 ring-blue-500/50'
                                : 'border-gray-700 hover:border-gray-600'
                            }`}
                    >
                        <img
                            src={img.url}
                            alt={img.name}
                            className="w-full h-32 object-cover"
                        />

                        {/* Overlay */}
                        <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    removeImage(index);
                                }}
                                className="p-2 bg-red-600 hover:bg-red-700 rounded-full"
                            >
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        </div>

                        {/* Selected Badge */}
                        {selectedImageIndex === index && (
                            <div className="absolute top-2 right-2 bg-blue-600 text-white text-xs px-2 py-1 rounded">
                                Active
                            </div>
                        )}

                        {/* Image Name */}
                        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-2">
                            <p className="text-xs text-white truncate">{img.name}</p>
                        </div>
                    </div>
                ))}
            </div>

            {targetImages.length === 0 && (
                <div className="text-center py-12 text-gray-500">
                    <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <p>No images uploaded</p>
                </div>
            )}
        </div>
    );
}

export default ImageUpload;
