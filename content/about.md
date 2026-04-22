---js
const eleventyNavigation = {
	key: "About",
	order: 2
};
---
# About

{{ metadata.description }}

<p>
  Want to talk machine learning or just say hi ? 
  <a href="mailto:{{ metadata.author.email }}">{{ metadata.author.email }}</a>
</p>
