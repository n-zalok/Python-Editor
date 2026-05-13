function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center py-8">
      <div className="h-12 w-12 animate-spin rounded-full border-4 border-cyan-400 border-t-transparent"></div>
    </div>
  );
}

export default LoadingSpinner;
