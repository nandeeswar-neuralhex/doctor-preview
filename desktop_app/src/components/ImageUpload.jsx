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

    return (
        <div className="space-y-3 md:space-y-4">
            <div>
                <h2 className="text-base md:text-lg font-semibold text-white mb-2">Target Images</h2>

                {/* Tips Box - hidden on mobile when images uploaded */}
                <div className={`bg-blue-900/20 border border-blue-500/30 rounded-lg p-2 md:p-3 mb-3 md:mb-4 ${targetImages.length > 0 ? 'hidden md:block' : ''}`}>
                    <div className="flex items-center gap-2 mb-1 md:mb-2 text-blue-400 font-medium text-xs md:text-sm">
                        <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        Tips for best results
                    </div>
                    <ul className="text-xs text-blue-200/80 space-y-1 list-disc list-inside">
                        <li>Use clear frontal photos (full face)</li>
                        <li>Upload varying angles and expressions</li>
                        <li>Ensure good lighting conditions</li>
                    </ul>
                </div>

                <p className="text-xs md:text-sm text-gray-400 mb-3 md:mb-4">
                    Upload 1-10 post-surgery preview images
                </p>
            </div>

            {/* Upload Button */}
            <button
                onClick={() => fileInputRef.current?.click()}
                disabled={targetImages.length >= 10 || !serverUrl}
                className="w-full py-2 md:py-3 px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2 text-sm md:text-base"
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

            {/* Image Grid - horizontal scroll on mobile, vertical stack on desktop */}
            <div className="flex md:flex-col gap-2 overflow-x-auto md:overflow-x-visible pb-2 md:pb-0">
                {targetImages.map((img, index) => (
                    <div
                        key={index}
                        onClick={() => setSelectedImageIndex(index)}
                        className={`relative cursor-pointer rounded-lg overflow-hidden border-2 transition-all flex-shrink-0 w-28 md:w-full ${selectedImageIndex === index
                            ? 'border-blue-500 ring-2 ring-blue-500/50'
                            : 'border-gray-700 hover:border-gray-600'
                            }`}
                    >
                        <img
                            src={img.url}
                            alt={img.name}
                            className="w-full h-20 md:h-32 object-cover"
                        />

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
                <div className="text-center py-6 md:py-12 text-gray-500">
                    <svg className="w-10 h-10 md:w-16 md:h-16 mx-auto mb-2 md:mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <p>No images uploaded</p>
                </div>
            )}
        </div>
    );
}

export default ImageUpload;
