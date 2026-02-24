/*=============== SHOW MENU ===============*/
const navMenu = document.getElementById('nav-menu'),
      navToggle = document.getElementById('nav-toggle'),
      navClose = document.getElementById('nav-close')
// Menu Show
if(navToggle){
    navToggle.addEventListener('click', () =>{
        navMenu.classList.add('show-menu');
        // animate menu sliding in
        gsap.from('.nav__menu', {x: 200, opacity: 0, duration: 0.5, ease: 'power2.out'});
    })
}

if(navClose){
    navClose.addEventListener('click', () =>{
        // animate menu sliding out then remove class
        gsap.to('.nav__menu', {x: 200, opacity: 0, duration: 0.5, ease: 'power2.in', onComplete: () => {
            navMenu.classList.remove('show-menu');
        }});
    })
}
/*=============== REMOVE MENU MOBILE ===============*/
const navLink = document.querySelectorAll('.nav__link')

function linkAction(){
    const navMenu = document.getElementById('nav-menu')
    // When we click on each nav__link, we remove the show-menu class
    navMenu.classList.remove('show-menu')
}
navLink.forEach(n => n.addEventListener('click', linkAction))
/*=============== GSAP ANIMATION ===============*/
gsap.from('.home__points',1.5,{opacity:0,delay:0.2,y:-300})
gsap.from('.home__rocket',1.5,{opacity:0,delay:0.3,y:300})
gsap.from('.home__planet-1',1.5,{opacity:0,delay:0.8,x:-200})
gsap.from('.home__planet-2',1.5,{opacity:0,delay:1.2,x:200})
gsap.from('.home__cloud-1',1.5,{opacity:0,delay:1.3,y:200})
gsap.from('.home__content',1.5,{opacity:0,delay:1.5,y:-100})
gsap.from('.home__title img',1.5,{opacity:0,delay:1.6,x:100})

// subtle infinite motion for background elements
gsap.to('.home__rocket',{y:-20,repeat:-1,yoyo:true,ease:'sine.inOut',duration:3});
gsap.to('.home__planet-1',{x:20,repeat:-1,yoyo:true,ease:'sine.inOut',duration:4});
gsap.to('.home__planet-2',{x:-20,repeat:-1,yoyo:true,ease:'sine.inOut',duration:4});
gsap.to('.home__cloud-1',{x:-30,repeat:-1,yoyo:true,ease:'sine.inOut',duration:6});
/* cloud-2 animation removed */

// when a nav link is clicked animate the rocket straight up then navigate
navLink.forEach(n => n.addEventListener('click', e => {
    e.preventDefault();
    const targetHref = n.getAttribute('href');
    linkAction(); // close mobile menu if open
    // fly rocket straight up
    gsap.to('.home__rocket',{x:0,y:-150,duration:1.5,ease:'power2.in',onComplete:()=>{
        // optional return or keep it out of view, no rotation
    }});
    // after delay, follow the link (simulate opening new page/file)
    setTimeout(() => {
        if(targetHref && targetHref !== '#')
            window.location = targetHref;
    }, 3000);
}));