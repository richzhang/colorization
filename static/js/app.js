document.addEventListener('DOMContentLoaded', function(){
  const form = document.getElementById('uploadForm');
  const fileInput = document.getElementById('imageInput');
  const loader = document.getElementById('loader');
  const carouselContainer = document.getElementById('carouselContainer');
  const carouselImg = document.getElementById('carouselImg');
  const caption = document.getElementById('caption');
  const prevBtn = document.getElementById('prevBtn');
  const nextBtn = document.getElementById('nextBtn');
  const thumbs = document.getElementById('thumbs');
  const thumbEls = document.querySelectorAll('.thumb');
  const btn = document.getElementById('colorizeBtn');
  const scores = document.getElementById('scores');

  let images = []; // urls in order: original, eccv, sig
  let idx = 1; // start with ECCV16 in center

  function showLoader(){ loader.style.display = 'flex'; loader.classList.remove('hidden'); loader.setAttribute('aria-hidden','false'); }
  function hideLoader(){ loader.style.display = 'none'; loader.classList.add('hidden'); loader.setAttribute('aria-hidden','true'); }

  function showCarousel(){
    carouselContainer.classList.remove('hidden');
    thumbs.classList.remove('hidden');
    scores.classList.remove('hidden');
  }

  function hideCarousel(){
    carouselContainer.classList.add('hidden');
    thumbs.classList.add('hidden');
    scores.classList.add('hidden');
  }

  function updateView(){
    carouselImg.src = images[idx] + '?t=' + Date.now();
    const labels = ['Original','ECCV16','SIGGRAPH17'];
    caption.textContent = labels[idx];
    thumbEls.forEach((t)=> t.classList.remove('active'));
    const a = document.querySelector('#thumb' + idx);
    if(a) a.classList.add('active');
  }

  prevBtn.addEventListener('click', function(){ idx = (idx - 1 + images.length) % images.length; updateView(); });
  nextBtn.addEventListener('click', function(){ idx = (idx + 1) % images.length; updateView(); });

  thumbEls.forEach((t)=>{
    t.addEventListener('click', function(e){
      const i = parseInt(t.getAttribute('data-index'));
      if(!isNaN(i)){ idx = i; updateView(); }
    });
  });

  // initial state - hide both
  hideCarousel();
  hideLoader();

  form.addEventListener('submit', function(e){
    e.preventDefault();
    if(!fileInput.files || fileInput.files.length===0){
      alert('Please select an image first');
      return;
    }

    // when starting a new upload hide any previous slideshow and show loader
    hideCarousel();
    showLoader();

    const fd = new FormData();
    fd.append('image', fileInput.files[0]);

    btn.disabled = true;

    fetch('/upload', {method:'POST', body: fd})
      .then(r=>r.json())
      .then(data=>{
        hideLoader();
        btn.disabled = false;
        if(data.success){
          images = [data.orig_url, data.eccv_url, data.sig_url];
          const eccvMetrics = data.scores.eccv16;
          const siggraphMetrics = data.scores.siggraph17;
          document.getElementById('eccvScore').textContent = eccvMetrics.similarity_score.toFixed(2);
          document.getElementById('eccvMetrics').innerHTML = formatMetrics(eccvMetrics);
          document.getElementById('sigScore').textContent = siggraphMetrics.similarity_score.toFixed(2);
          document.getElementById('sigMetrics').innerHTML = formatMetrics(siggraphMetrics);
          // populate thumbs
          document.getElementById('thumb0').src = images[0] + '?t=' + Date.now();
          document.getElementById('thumb1').src = images[1] + '?t=' + Date.now();
          document.getElementById('thumb2').src = images[2] + '?t=' + Date.now();

          idx = 1; // center on ECCV16
          updateView();
          showCarousel();
        } else {
          alert('Error: ' + (data.error || 'unknown'));
        }
      })
      .catch(err=>{
        hideLoader();
        btn.disabled = false;
        alert('Upload failed');
        console.error(err);
      });
  });

  function formatMetrics(metrics){
    const psnr = metrics.psnr === null ? '∞' : metrics.psnr.toFixed(4) + ' dB';
    return 'CIEDE2000: ' + metrics.mean_delta_e.toFixed(4) +
      '<br>MSE: ' + metrics.mse.toFixed(8) +
      '<br>PSNR: ' + psnr +
      '<br>SSIM: ' + metrics.ssim.toFixed(4);
  }
});
