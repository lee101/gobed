package search

import "cmp"

// scored is a candidate score with its position in the candidate slice.
// Higher s is better; ties prefer the lower pos.
type scored[T cmp.Ordered] struct {
	s   T
	pos int32
}

func worse[T cmp.Ordered](a, b scored[T]) bool {
	return a.s < b.s || (a.s == b.s && a.pos > b.pos)
}

// topRPush keeps the best R items in h as a heap rooted at the worst item.
// Items must be pushed in increasing pos order.
func topRPush[T cmp.Ordered](h []scored[T], R int, x scored[T]) []scored[T] {
	if len(h) < R {
		h = append(h, x)
		i := len(h) - 1
		for i > 0 {
			p := (i - 1) / 2
			if !worse(h[i], h[p]) {
				break
			}
			h[i], h[p] = h[p], h[i]
			i = p
		}
		return h
	}
	if R <= 0 || x.s <= h[0].s {
		return h
	}
	h[0] = x
	siftDown(h, 0)
	return h
}

func siftDown[T cmp.Ordered](h []scored[T], i int) {
	n := len(h)
	for {
		l := 2*i + 1
		if l >= n {
			return
		}
		m := l
		if r := l + 1; r < n && worse(h[r], h[l]) {
			m = r
		}
		if !worse(h[m], h[i]) {
			return
		}
		h[i], h[m] = h[m], h[i]
		i = m
	}
}

// topRSort heap-sorts h in place into best-first order.
func topRSort[T cmp.Ordered](h []scored[T]) {
	for end := len(h) - 1; end > 0; end-- {
		h[0], h[end] = h[end], h[0]
		siftDown(h[:end], 0)
	}
}
