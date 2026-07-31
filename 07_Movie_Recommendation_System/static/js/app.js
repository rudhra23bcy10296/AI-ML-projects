/*
  Project 7: TMDB Movie Recommendation System Interactive JS Engine
  Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
*/

function initApp() {
  const searchInput = document.getElementById('searchInput');
  const autocompleteList = document.getElementById('autocompleteList');
  const btnSearch = document.getElementById('btnSearch');
  const heroBanner = document.getElementById('heroBanner');
  const moviesGrid = document.getElementById('moviesGrid');

  let debounceTimer = null;

  // Autocomplete Search Debounce Listener
  if (searchInput && autocompleteList) {
    searchInput.addEventListener('input', (e) => {
      clearTimeout(debounceTimer);
      const query = e.target.value.trim();
      
      if (query.length === 0) {
        autocompleteList.style.display = 'none';
        return;
      }

      debounceTimer = setTimeout(async () => {
        try {
          const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
          const data = await res.json();
          if (data.status === 'success' && data.results.length > 0) {
            autocompleteList.innerHTML = '';
            data.results.forEach(title => {
              const item = document.createElement('div');
              item.className = 'autocomplete-item';
              item.textContent = title;
              item.addEventListener('click', () => {
                searchInput.value = title;
                autocompleteList.style.display = 'none';
                fetchRecommendations(title);
              });
              autocompleteList.appendChild(item);
            });
            autocompleteList.style.display = 'block';
          } else {
            autocompleteList.style.display = 'none';
          }
        } catch (err) {
          console.error('Autocomplete search error:', err);
        }
      }, 250);
    });

    document.addEventListener('click', (e) => {
      if (!searchInput.contains(e.target) && !autocompleteList.contains(e.target)) {
        autocompleteList.style.display = 'none';
      }
    });
  }

  // Fetch Recommendations Function
  async function fetchRecommendations(title) {
    if (!title || title.trim().length === 0) return;

    if (btnSearch) {
      btnSearch.disabled = true;
      btnSearch.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Calculating Vector Distances...';
    }

    try {
      const res = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: title.trim(), top_n: 6 })
      });

      const data = await res.json();

      if (data.status === 'success') {
        const qm = data.query_movie;
        const recs = data.recommendations;

        // Render Query Movie Hero Banner
        if (heroBanner) {
          document.getElementById('heroPoster').src = qm.poster;
          document.getElementById('heroTitle').textContent = qm.title;
          document.getElementById('heroYear').textContent = qm.year;
          document.getElementById('heroRating').textContent = `${qm.vote_average} ⭐`;
          heroBanner.style.display = 'flex';
        }

        // Render Recommendation Cards
        if (moviesGrid) {
          moviesGrid.innerHTML = '';
          recs.forEach(m => {
            moviesGrid.innerHTML += `
              <div class="movie-card">
                <div class="poster-wrapper">
                  <img src="${m.poster}" class="card-poster" alt="${m.title}" loading="lazy">
                  <div class="score-tag"><i class="fas fa-bolt"></i> ${m.similarity_score}% Match</div>
                </div>
                <div class="card-details">
                  <div class="card-title">${m.title}</div>
                  <div class="card-meta">
                    <span><i class="far fa-calendar"></i> ${m.year}</span>
                    <span class="rating-star"><i class="fas fa-star"></i> ${m.vote_average}</span>
                  </div>
                </div>
              </div>
            `;
          });
        }
      }
    } catch (err) {
      console.error('Fetch recommendations error:', err);
    } finally {
      if (btnSearch) {
        btnSearch.disabled = false;
        btnSearch.innerHTML = '<i class="fas fa-magic"></i> Get Recommendations';
      }
    }
  }

  if (btnSearch && searchInput) {
    btnSearch.addEventListener('click', () => {
      fetchRecommendations(searchInput.value);
    });

    searchInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        autocompleteList.style.display = 'none';
        fetchRecommendations(searchInput.value);
      }
    });
  }

  // Auto-fetch recommendations on page load
  fetchRecommendations('Inception');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
